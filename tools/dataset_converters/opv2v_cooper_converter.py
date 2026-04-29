# Copyright (c) OpenMMLab. All rights reserved.
"""OPV2V Cooperative Dataset Converter.

Converts OPV2V dataset to cooperative (multi-agent) mmdetection3d format.
Each (scenario, timestamp) group produces N samples where each agent takes
a turn as ego, following the OpenCOOD/CoBEVT convention.

Usage:
    python tools/dataset_converters/opv2v_cooper_converter.py \
        --data-root data/opv2v \
        --out-dir data/opv2v \
        --convert-pcd  # optional: convert .pcd to .bin
"""

import argparse
import os
from collections import defaultdict
from concurrent import futures
from os import path as osp
from typing import List

import mmengine
import numpy as np
from tqdm import tqdm

from opv2v_converter import (
    CARLA_TO_CV_CAM, convert_pcd_to_bin, parse_opv2v_vehicle)


def extract_camera_info(meta, data_root, split, scenario, agent_id,
                        timestamp):
    """Extract camera information from agent metadata.

    Reuses the same camera parsing logic as opv2v_converter.py for
    consistency between single-agent and cooperative datasets.

    Args:
        meta: Parsed YAML metadata dict for this agent.
        data_root: Root path of OPV2V dataset.
        split: Dataset split name (train/validate/test).
        scenario: Scenario directory name.
        agent_id: Agent directory name.
        timestamp: Timestamp string.

    Returns:
        Dict mapping camera names to dicts with img_path, lidar2cam,
        cam2img, or empty dict if no cameras found.
    """
    images = {}
    for cam_id in ['camera0', 'camera1', 'camera2', 'camera3']:
        if cam_id not in meta:
            continue
        cam_data = meta[cam_id]

        cam_img_filename = f'{timestamp}_{cam_id}.png'
        cam_img_path = osp.join(
            data_root, split, scenario, agent_id, cam_img_filename)
        if not osp.exists(cam_img_path):
            continue

        extrinsic_carla = np.array(
            cam_data['extrinsic'], dtype=np.float64)
        lidar2cam = (CARLA_TO_CV_CAM @ extrinsic_carla).astype(
            np.float32)

        cam2img = np.array(
            cam_data['intrinsic'], dtype=np.float32)

        images[cam_id] = {
            'img_path': osp.join(
                split, scenario, agent_id, cam_img_filename),
            'lidar2cam': lidar2cam.tolist(),
            'cam2img': cam2img.tolist(),
        }

    return images


def create_cooperative_infos(
    data_root: str,
    out_dir: str,
    splits: List[str] = ['train', 'validate', 'test'],
    convert_pcd: bool = False,
    num_workers: int = 8
) -> None:
    """Create cooperative info files for OPV2V dataset.

    Groups agents by (scenario, timestamp). Each agent takes a turn as
    ego, producing N samples per group (round-robin). All other agents
    in the group become cooperators with their poses and LiDAR paths.

    Args:
        data_root: Root path of OPV2V dataset.
        out_dir: Output directory for pkl files.
        splits: List of splits to process.
        convert_pcd: Whether to convert .pcd to .bin files.
        num_workers: Number of workers for parallel processing.
    """
    import yaml

    split_mapping = {
        'train': 'train',
        'validate': 'val',
        'test': 'test',
        'test_culvercity': 'test_culvercity'
    }

    os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        split_path = osp.join(data_root, split)
        if not osp.exists(split_path):
            print(f'Skip {split}, not found at {split_path}')
            continue

        print(f'\nProcessing cooperative {split}...')

        # Collect all yaml files grouped by (scenario, timestamp)
        groups = defaultdict(list)
        scenarios = [d for d in os.listdir(split_path)
                     if osp.isdir(osp.join(split_path, d))]

        for scenario in scenarios:
            scenario_path = osp.join(split_path, scenario)
            agents = [d for d in os.listdir(scenario_path)
                      if osp.isdir(osp.join(scenario_path, d))]

            for agent_id in agents:
                agent_path = osp.join(scenario_path, agent_id)
                for f in os.listdir(agent_path):
                    if (f.endswith('.yaml') and
                            not f.startswith('data_protocol') and
                            not f.endswith('_additional.yaml')):
                        timestamp = f.replace('.yaml', '')
                        yaml_path = osp.join(agent_path, f)
                        groups[(scenario, timestamp)].append(
                            (yaml_path, agent_id))

        print(f'Found {len(groups)} (scenario, timestamp) groups')

        # Convert PCD to BIN if requested
        if convert_pcd:
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
                with futures.ThreadPoolExecutor(
                        max_workers=num_workers) as executor:
                    list(tqdm(
                        executor.map(
                            lambda x: convert_pcd_to_bin(x[0], x[1]),
                            pcd_files),
                        total=len(pcd_files),
                        desc='Converting'))

        # Process each group
        infos = []
        sample_idx = 0

        for (scenario, timestamp), agents in tqdm(
                sorted(groups.items()),
                desc='Processing cooperative annotations'):

            # Parse all agents' metadata
            agent_data = []
            for yaml_path, agent_id in agents:
                try:
                    with open(yaml_path, 'r') as f:
                        meta = yaml.unsafe_load(f)
                except Exception:
                    continue

                ego_pose = meta.get('lidar_pose', meta.get('true_ego_pos'))
                if ego_pose is None:
                    continue

                # Determine point cloud path
                bin_path = osp.join(
                    data_root, split, scenario, agent_id,
                    f'{timestamp}.bin')
                if osp.exists(bin_path):
                    pts_filename = f'{timestamp}.bin'
                else:
                    pts_filename = f'{timestamp}.pcd'

                pts_path = osp.join(
                    data_root, split, scenario, agent_id, pts_filename)
                if not osp.exists(pts_path):
                    continue

                # Extract camera info for this agent
                images = extract_camera_info(
                    meta, data_root, split, scenario, agent_id,
                    timestamp)

                agent_data.append({
                    'agent_id': agent_id,
                    'ego_pose': ego_pose,
                    'lidar_path': osp.join(
                        split, scenario, agent_id, pts_filename),
                    'meta': meta,
                    'images': images,
                })

            if not agent_data:
                continue

            # Each agent takes a turn as ego, producing one sample each.
            # This follows the OpenCOOD/CoBEVT convention and multiplies
            # the dataset size by the number of agents per group.
            agent_data.sort(key=lambda x: x['agent_id'])

            for ego_idx, ego in enumerate(agent_data):
                cooperators = [a for j, a in enumerate(agent_data)
                               if j != ego_idx]

                # Parse GT annotations from ego's perspective
                instances = []
                vehicles = ego['meta'].get('vehicles', {})
                if vehicles:
                    for veh_id, veh_data in vehicles.items():
                        bbox_ego = parse_opv2v_vehicle(
                            veh_data, ego['ego_pose'])
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

                # Build cooperative info
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
                }

                # Add ego camera images (always include, like cooperators)
                coop_info['images'] = ego['images']

                infos.append(coop_info)
                sample_idx += 1

        print(f'Processed {len(infos)} cooperative samples')

        # Save pkl file
        out_split = split_mapping.get(split, split)
        output = {
            'data_list': infos,
            'metainfo': {
                'classes': ('vehicle',),
                'categories': {'vehicle': 0},
                'dataset': 'OPV2V',
                'info_version': '1.0',
                'cooperative': True,
            }
        }
        out_path = osp.join(out_dir, f'opv2v_coop_infos_{out_split}.pkl')
        mmengine.dump(output, out_path)
        print(f'Saved {len(infos)} cooperative samples to {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='OPV2V Cooperative Dataset Converter')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/opv2v',
        help='Root path of OPV2V dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/opv2v',
        help='Output directory for pkl files')
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validate', 'test'],
        help='Splits to process')
    parser.add_argument(
        '--convert-pcd',
        action='store_true',
        help='Convert .pcd files to .bin files')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of workers for parallel processing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    create_cooperative_infos(
        data_root=args.data_root,
        out_dir=args.out_dir,
        splits=args.splits,
        convert_pcd=args.convert_pcd,
        num_workers=args.num_workers,
    )
