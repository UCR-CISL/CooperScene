#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""CooperScene Dataset Converter.

Converts CooperScene dataset to mmdetection3d format.
Handles 4x4 matrix lidar_pose and flat directory structure
(no train/val/test split directories).

Dataset structure:
    data_root/
    └── scenario_id/
        └── agent_id/
            ├── {timestamp}_separate.yaml
            ├── {timestamp}.pcd
            ├── {timestamp}_camera0.png
            └── metadata.json

Usage:
    # Single-agent only:
    python tools/dataset_converters/cooperscene_converter.py \
        --data-root /path/to/cooperscene \
        --out-dir /path/to/cooperscene \
        --val-ratio 0.2 \
        --convert-pcd

    # Both single-agent and cooperative:
    python tools/dataset_converters/cooperscene_converter.py \
        --data-root /path/to/cooperscene \
        --out-dir /path/to/cooperscene \
        --val-ratio 0.2 \
        --cooperative \
        --convert-pcd
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent import futures
from os import path as osp
from typing import Dict, List, Optional

import mmengine
import numpy as np
from tqdm import tqdm

from opv2v_converter import convert_pcd_to_bin, get_rotation_matrix


# Global calibration data loaded once
_CALIBRATION = None


def load_calibration(calibration_file: str) -> Dict:
    """Load calibration results from JSON file.

    Args:
        calibration_file: Path to calibration_results.json.

    Returns:
        Dict mapping (scenario, agent_id) -> {T_lidar_to_cam, intrinsic}.
    """
    with open(calibration_file, 'r') as f:
        raw = json.load(f)

    calib = {}
    for key, val in raw.items():
        # Key format: "scene{scenario}_agent{agent_id}"
        parts = key.split('_')
        scenario = parts[0].replace('scene', '')
        agent_id = parts[1].replace('agent', '')
        calib[(scenario, agent_id)] = {
            'T_lidar_to_cam': np.array(
                val['T_lidar_to_cam'], dtype=np.float32),
            'intrinsic': np.array(val['intrinsic'], dtype=np.float32),
        }
    return calib


def get_camera_info(scenario, agent_id, timestamp, path_prefix=''):
    """Get camera image info dict for a given frame.

    Args:
        scenario: Scenario ID string.
        agent_id: Agent ID string.
        timestamp: Timestamp string.
        path_prefix: Prefix for relative paths (e.g. 'train').

    Returns:
        Dict with camera0 info, or empty dict if no calibration.
    """
    global _CALIBRATION
    if _CALIBRATION is None:
        return {}

    key = (scenario, agent_id)
    if key not in _CALIBRATION:
        return {}

    rel_base = osp.join(path_prefix, scenario, agent_id) \
        if path_prefix else osp.join(scenario, agent_id)

    cal = _CALIBRATION[key]
    return {
        'camera0': {
            'img_path': osp.join(
                rel_base, f'{timestamp}_camera0.png'),
            'lidar2cam': cal['T_lidar_to_cam'].tolist(),
            'cam2img': cal['intrinsic'].tolist(),
        }
    }


def transform_world_to_ego_matrix(
    bbox_world: List[float],
    ego_matrix: np.ndarray
) -> List[float]:
    """Transform bounding box from world to ego coordinates using 4x4 matrix.

    Args:
        bbox_world: [x, y, z, dx, dy, dz, yaw_rad] in world frame.
        ego_matrix: 4x4 transformation matrix (ego lidar pose in world frame).

    Returns:
        [x, y, z, dx, dy, dz, yaw] in ego-lidar frame.
    """
    T_ego_inv = np.linalg.inv(ego_matrix)

    # Transform position
    pos_world = np.array(
        [bbox_world[0], bbox_world[1], bbox_world[2], 1.0])
    pos_ego = (T_ego_inv @ pos_world)[:3]

    # Size stays the same
    size = bbox_world[3:6]

    # Transform yaw: rotate heading vector to ego frame
    yaw_world = bbox_world[6]
    heading_world = np.array(
        [np.cos(yaw_world), np.sin(yaw_world), 0.0])
    heading_ego = T_ego_inv[:3, :3] @ heading_world
    yaw_ego = float(np.arctan2(heading_ego[1], heading_ego[0]))

    return [
        float(pos_ego[0]), float(pos_ego[1]), float(pos_ego[2]),
        float(size[0]), float(size[1]), float(size[2]),
        yaw_ego
    ]


def parse_vehicle(
    veh_data: Dict,
    ego_matrix: np.ndarray
) -> Optional[List[float]]:
    """Convert vehicle annotation to mmdet3d format using 4x4 ego matrix.

    Args:
        veh_data: Vehicle annotation dict from yaml.
        ego_matrix: 4x4 ego lidar pose matrix in world frame.

    Returns:
        Bounding box [x, y, z, dx, dy, dz, yaw] in ego coords, or None.
    """
    try:
        loc = veh_data['location']       # [x, y, z] world frame
        extent = veh_data['extent']      # [half_l, half_w, half_h]
        angle = veh_data['angle']        # [roll, yaw, pitch] degrees
        center = veh_data.get('center', [0, 0, 0])

        # Full size (mmdet3d uses full size, not half)
        dx = extent[0] * 2
        dy = extent[1] * 2
        dz = extent[2] * 2

        # Rotate center offset by vehicle's orientation
        R_veh = get_rotation_matrix(angle[0], angle[1], angle[2])
        center_world = R_veh @ np.array(center)

        # Actual bbox center in world frame
        bbox_x = loc[0] + center_world[0]
        bbox_y = loc[1] + center_world[1]
        bbox_z = loc[2] + center_world[2]

        yaw_world = np.radians(angle[1])
        bbox_world = [bbox_x, bbox_y, bbox_z, dx, dy, dz, yaw_world]

        return transform_world_to_ego_matrix(bbox_world, ego_matrix)
    except (KeyError, IndexError, TypeError) as e:
        print(f'Warning: Failed to parse vehicle: {e}')
        return None


def parse_lidar_pose_matrix(meta: Dict) -> Optional[np.ndarray]:
    """Parse lidar_pose from yaml metadata as a 4x4 matrix.

    Handles both formats:
    - 4x4 matrix (list of 4 lists of 4 floats) — CooperScene format
    - 6-element list [x, y, z, roll, yaw, pitch] — OPV2V format

    Args:
        meta: Parsed yaml dict.

    Returns:
        4x4 numpy array or None if not found.
    """
    pose = meta.get('lidar_pose', meta.get('true_ego_pos'))
    if pose is None:
        return None

    pose_arr = np.array(pose, dtype=np.float64)
    if pose_arr.shape == (4, 4):
        return pose_arr
    elif pose_arr.shape == (6,):
        # Convert 6-element to 4x4 matrix
        x, y, z = pose_arr[0], pose_arr[1], pose_arr[2]
        R = get_rotation_matrix(pose_arr[3], pose_arr[4], pose_arr[5])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    else:
        print(f'Warning: Unexpected lidar_pose shape: {pose_arr.shape}')
        return None


def collect_dataset_files(data_root: str, split_dir: str = None):
    """Collect all yaml files from the directory structure.

    Supports two layouts:
    - Flat: data_root/scenario/agent/{timestamp}_separate.yaml
    - Split: data_root/{split}/scenario/agent/{timestamp}_separate.yaml

    Args:
        data_root: Root path of dataset.
        split_dir: If given, scan data_root/{split_dir}/ for scenarios.
                   If None, scan data_root/ directly.

    Returns:
        Dict mapping (scenario, timestamp) -> list of (yaml_path, agent_id).
    """
    groups = defaultdict(list)

    scan_root = osp.join(data_root, split_dir) if split_dir else data_root

    scenarios = sorted([
        d for d in os.listdir(scan_root)
        if osp.isdir(osp.join(scan_root, d))
        and not d.endswith('_infra')
    ])

    for scenario in scenarios:
        scenario_path = osp.join(scan_root, scenario)
        agents = sorted([
            d for d in os.listdir(scenario_path)
            if osp.isdir(osp.join(scenario_path, d))
        ])

        for agent_id in agents:
            agent_path = osp.join(scenario_path, agent_id)
            for f in os.listdir(agent_path):
                if (f.endswith('_separate.yaml') and
                        not f.startswith('data_protocol')):
                    timestamp = f.replace('_separate.yaml', '')
                    yaml_path = osp.join(agent_path, f)
                    groups[(scenario, timestamp)].append(
                        (yaml_path, agent_id))

    return groups


def split_by_timestamp(groups, val_ratio=0.2):
    """Split groups into train/val by timestamp within each scenario.

    Args:
        groups: Dict from collect_dataset_files.
        val_ratio: Fraction of frames for validation.

    Returns:
        (train_groups, val_groups) dicts.
    """
    # Group timestamps by scenario
    scenario_timestamps = defaultdict(set)
    for (scenario, timestamp) in groups:
        scenario_timestamps[scenario].add(timestamp)

    # Determine split point per scenario
    train_keys = set()
    val_keys = set()
    for scenario, timestamps in scenario_timestamps.items():
        sorted_ts = sorted(timestamps, key=int)
        n_val = max(1, int(len(sorted_ts) * val_ratio))
        n_train = len(sorted_ts) - n_val

        for ts in sorted_ts[:n_train]:
            train_keys.add((scenario, ts))
        for ts in sorted_ts[n_train:]:
            val_keys.add((scenario, ts))

    train_groups = {k: v for k, v in groups.items() if k in train_keys}
    val_groups = {k: v for k, v in groups.items() if k in val_keys}

    return train_groups, val_groups


def process_single_agent(groups, scan_root, path_prefix=''):
    """Generate single-agent info dicts from grouped files.

    Args:
        groups: Dict mapping (scenario, timestamp) -> [(yaml_path, agent_id)].
        scan_root: Absolute path where scenario dirs live (for file lookups).
        path_prefix: Prefix for relative paths stored in pkl (e.g. 'train').

    Returns:
        List of info dicts.
    """
    import yaml

    infos = []
    sample_idx = 0

    for (scenario, timestamp), agents in tqdm(
            sorted(groups.items()),
            desc='Processing single-agent'):

        for yaml_path, agent_id in agents:
            try:
                with open(yaml_path, 'r') as f:
                    meta = yaml.unsafe_load(f)
            except Exception as e:
                print(f'Error loading {yaml_path}: {e}')
                continue

            # Parse ego pose as 4x4 matrix
            ego_matrix = parse_lidar_pose_matrix(meta)
            if ego_matrix is None:
                continue

            # Determine point cloud path
            bin_path = osp.join(
                scan_root, scenario, agent_id, f'{timestamp}.bin')
            if osp.exists(bin_path):
                pts_filename = f'{timestamp}.bin'
            else:
                pts_filename = f'{timestamp}.pcd'

            pts_path = osp.join(
                scan_root, scenario, agent_id, pts_filename)
            if not osp.exists(pts_path):
                continue

            # Relative path: path_prefix/scenario/agent/file
            rel_base = osp.join(path_prefix, scenario, agent_id) \
                if path_prefix else osp.join(scenario, agent_id)

            # Store ego_pose as 4x4 matrix (list of lists for pkl)
            info = {
                'sample_idx': sample_idx,
                'lidar_points': {
                    'lidar_path': osp.join(rel_base, pts_filename),
                    'num_pts_feats': 4,
                },
                'images': get_camera_info(
                    scenario, agent_id, timestamp, path_prefix),
                'timestamp': float(timestamp),
                'scenario': scenario,
                'agent_id': agent_id,
                'ego_pose': ego_matrix.tolist(),
                'instances': [],
            }

            # Parse vehicle annotations
            vehicles = meta.get('vehicles', {})
            if vehicles:
                for veh_id, veh_data in vehicles.items():
                    bbox_ego = parse_vehicle(veh_data, ego_matrix)
                    if bbox_ego is None:
                        continue
                    instance = {
                        'bbox_3d': bbox_ego,
                        'bbox_label_3d': 0,
                    }
                    if 'speed' in veh_data:
                        speed = veh_data['speed'] / 3.6
                        yaw = bbox_ego[6]
                        vx = speed * np.cos(yaw)
                        vy = speed * np.sin(yaw)
                        instance['velocity'] = [float(vx), float(vy)]
                    info['instances'].append(instance)

            infos.append(info)
            sample_idx += 1

    return infos


def process_cooperative(groups, scan_root, path_prefix=''):
    """Generate cooperative info dicts from grouped files.

    Each (scenario, timestamp) group produces N samples where each agent
    takes a turn as ego (round-robin).

    Args:
        groups: Dict mapping (scenario, timestamp) -> [(yaml_path, agent_id)].
        scan_root: Absolute path where scenario dirs live (for file lookups).
        path_prefix: Prefix for relative paths stored in pkl (e.g. 'train').

    Returns:
        List of cooperative info dicts.
    """
    import yaml

    infos = []
    sample_idx = 0

    for (scenario, timestamp), agents in tqdm(
            sorted(groups.items()),
            desc='Processing cooperative'):

        # Parse all agents' metadata
        agent_data = []
        for yaml_path, agent_id in agents:
            try:
                with open(yaml_path, 'r') as f:
                    meta = yaml.unsafe_load(f)
            except Exception:
                continue

            ego_matrix = parse_lidar_pose_matrix(meta)
            if ego_matrix is None:
                continue

            # Determine point cloud path
            bin_path = osp.join(
                scan_root, scenario, agent_id, f'{timestamp}.bin')
            if osp.exists(bin_path):
                pts_filename = f'{timestamp}.bin'
            else:
                pts_filename = f'{timestamp}.pcd'

            pts_path = osp.join(
                scan_root, scenario, agent_id, pts_filename)
            if not osp.exists(pts_path):
                continue

            rel_base = osp.join(path_prefix, scenario, agent_id) \
                if path_prefix else osp.join(scenario, agent_id)

            agent_data.append({
                'agent_id': agent_id,
                'ego_matrix': ego_matrix,
                'ego_pose': ego_matrix.tolist(),
                'lidar_path': osp.join(rel_base, pts_filename),
                'meta': meta,
                'images': get_camera_info(
                    scenario, agent_id, timestamp, path_prefix),
            })

        if not agent_data:
            continue

        # Sort by agent_id for deterministic ordering
        agent_data.sort(key=lambda x: x['agent_id'])

        # Each agent takes a turn as ego
        for ego_idx, ego in enumerate(agent_data):
            cooperators = [a for j, a in enumerate(agent_data)
                           if j != ego_idx]

            # Parse GT annotations from ego's perspective
            instances = []
            vehicles = ego['meta'].get('vehicles', {})
            if vehicles:
                for veh_id, veh_data in vehicles.items():
                    bbox_ego = parse_vehicle(
                        veh_data, ego['ego_matrix'])
                    if bbox_ego is None:
                        continue
                    instance = {
                        'bbox_3d': bbox_ego,
                        'bbox_label_3d': 0,
                    }
                    if 'speed' in veh_data:
                        speed = veh_data['speed'] / 3.6
                        yaw = bbox_ego[6]
                        vx = speed * np.cos(yaw)
                        vy = speed * np.sin(yaw)
                        instance['velocity'] = [float(vx), float(vy)]
                    instances.append(instance)

            coop_info = {
                'sample_idx': sample_idx,
                'scenario': scenario,
                'timestamp': float(timestamp),
                'agent_id': ego['agent_id'],
                'ego_pose': ego['ego_pose'],
                'lidar_points': {
                    'lidar_path': ego['lidar_path'],
                    'num_pts_feats': 4,
                },
                'instances': instances,
                'cooperators': [
                    {
                        'agent_id': coop['agent_id'],
                        'ego_pose': coop['ego_pose'],
                        'lidar_points': {
                            'lidar_path': coop['lidar_path'],
                            'num_pts_feats': 4,
                        },
                        'images': coop['images'],
                    }
                    for coop in cooperators
                ],
                'images': ego['images'],
            }

            infos.append(coop_info)
            sample_idx += 1

    return infos


def save_pkl(infos, out_path, cooperative=False):
    """Save info list to pkl file.

    Args:
        infos: List of info dicts.
        out_path: Output pkl path.
        cooperative: Whether this is cooperative format.
    """
    output = {
        'data_list': infos,
        'metainfo': {
            'classes': ('vehicle',),
            'categories': {'vehicle': 0},
            'dataset': 'CooperScene',
            'info_version': '1.0',
        }
    }
    if cooperative:
        output['metainfo']['cooperative'] = True

    mmengine.dump(output, out_path)
    print(f'Saved {len(infos)} samples to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='CooperScene Dataset Converter')
    parser.add_argument(
        '--data-root', type=str, required=True,
        help='Root path of CooperScene dataset (e.g., .../250928_opv2v)')
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory for pkl files (default: same as data-root)')
    parser.add_argument(
        '--splits', nargs='+', default=None,
        help='Predefined split dirs under data-root (e.g., train validate test). '
             'If not set, uses flat layout with --val-ratio.')
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Fraction of frames for validation (flat layout only)')
    parser.add_argument(
        '--cooperative', action='store_true',
        help='Also generate cooperative pkl files')
    parser.add_argument(
        '--convert-pcd', action='store_true',
        help='Convert .pcd files to .bin files')
    parser.add_argument(
        '--calibration-file', type=str, default=None,
        help='Path to calibration_results.json for camera data')
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='Number of workers for PCD conversion')
    args = parser.parse_args()

    data_root = args.data_root
    out_dir = args.out_dir or data_root
    os.makedirs(out_dir, exist_ok=True)

    # Load camera calibration if provided
    global _CALIBRATION
    if args.calibration_file:
        print(f'Loading calibration from {args.calibration_file}')
        _CALIBRATION = load_calibration(args.calibration_file)
        print(f'  Loaded calibration for {len(_CALIBRATION)} agents')

    # Mapping from split name to pkl suffix
    split_to_suffix = {
        'train': 'train',
        'validate': 'val',
        'val': 'val',
        'test': 'test',
    }

    if args.splits:
        # ---- Predefined split directories ----
        for split in args.splits:
            split_path = osp.join(data_root, split)
            if not osp.isdir(split_path):
                print(f'Warning: split dir {split_path} not found, skipping')
                continue

            suffix = split_to_suffix.get(split, split)
            print(f'\n=== Collecting {split} ===')
            groups = collect_dataset_files(data_root, split_dir=split)
            print(f'Found {len(groups)} (scenario, timestamp) groups')

            scan_root = osp.join(data_root, split)

            # Convert PCD to BIN if requested
            if args.convert_pcd:
                pcd_files = []
                for (scenario, timestamp), agents in groups.items():
                    for yaml_path, agent_id in agents:
                        agent_path = osp.dirname(yaml_path)
                        pcd_path = osp.join(
                            agent_path, f'{timestamp}.pcd')
                        bin_path = osp.join(
                            agent_path, f'{timestamp}.bin')
                        if osp.exists(pcd_path) and not osp.exists(bin_path):
                            pcd_files.append((pcd_path, bin_path))
                if pcd_files:
                    print(f'Converting {len(pcd_files)} PCD files...')
                    with futures.ThreadPoolExecutor(
                            max_workers=args.num_workers) as executor:
                        list(tqdm(
                            executor.map(
                                lambda x: convert_pcd_to_bin(x[0], x[1]),
                                pcd_files),
                            total=len(pcd_files),
                            desc='Converting'))

            # Single-agent
            print(f'\n--- Single-Agent ({split}) ---')
            infos = process_single_agent(
                groups, scan_root, path_prefix=split)
            save_pkl(infos, osp.join(
                out_dir, f'cooperscene_infos_{suffix}.pkl'))

            # Cooperative
            if args.cooperative:
                print(f'\n--- Cooperative ({split}) ---')
                coop_infos = process_cooperative(
                    groups, scan_root, path_prefix=split)
                save_pkl(coop_infos, osp.join(
                    out_dir, f'cooperscene_coop_infos_{suffix}.pkl'),
                    cooperative=True)
    else:
        # ---- Flat layout with temporal split ----
        print('Collecting dataset files...')
        groups = collect_dataset_files(data_root)
        print(f'Found {len(groups)} (scenario, timestamp) groups')

        # Convert PCD to BIN if requested
        if args.convert_pcd:
            print('Converting PCD to BIN...')
            pcd_files = []
            for (scenario, timestamp), agents in groups.items():
                for yaml_path, agent_id in agents:
                    agent_path = osp.dirname(yaml_path)
                    pcd_path = osp.join(agent_path, f'{timestamp}.pcd')
                    bin_path = osp.join(agent_path, f'{timestamp}.bin')
                    if osp.exists(pcd_path) and not osp.exists(bin_path):
                        pcd_files.append((pcd_path, bin_path))

            if pcd_files:
                print(f'Converting {len(pcd_files)} PCD files...')
                with futures.ThreadPoolExecutor(
                        max_workers=args.num_workers) as executor:
                    list(tqdm(
                        executor.map(
                            lambda x: convert_pcd_to_bin(x[0], x[1]),
                            pcd_files),
                        total=len(pcd_files),
                        desc='Converting'))

        # Split into train/val
        train_groups, val_groups = split_by_timestamp(
            groups, args.val_ratio)
        print(f'Train: {len(train_groups)} groups, '
              f'Val: {len(val_groups)} groups')

        # Generate single-agent pkl files
        print('\n=== Single-Agent ===')
        train_infos = process_single_agent(train_groups, data_root)
        save_pkl(train_infos,
                 osp.join(out_dir, 'cooperscene_infos_train.pkl'))

        val_infos = process_single_agent(val_groups, data_root)
        save_pkl(val_infos,
                 osp.join(out_dir, 'cooperscene_infos_val.pkl'))

        # Generate cooperative pkl files
        if args.cooperative:
            print('\n=== Cooperative ===')
            train_coop = process_cooperative(train_groups, data_root)
            save_pkl(train_coop,
                     osp.join(out_dir, 'cooperscene_coop_infos_train.pkl'),
                     cooperative=True)

            val_coop = process_cooperative(val_groups, data_root)
            save_pkl(val_coop,
                     osp.join(out_dir, 'cooperscene_coop_infos_val.pkl'),
                     cooperative=True)

    print('\nDone!')


if __name__ == '__main__':
    main()
