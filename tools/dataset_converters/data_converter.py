# Copyright (c) OpenMMLab. All rights reserved.
"""CooperScene Dataset Converter.

This script converts CooperScene dataset to mmdetection3d format.

Usage:
    python tools/dataset_converters/data_converter.py \
        --data-root data/cooperscene \
        --out-dir data/cooperscene \
        --convert-pcd  # optional: convert .pcd to .bin
"""

import argparse
import os
from concurrent import futures
from os import path as osp
from typing import Dict, List, Optional, Tuple

import mmengine
import numpy as np
from tqdm import tqdm


def get_rotation_matrix(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """Get rotation matrix from Euler angles (in degrees).

    Args:
        roll: Roll angle in degrees.
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.

    Returns:
        3x3 rotation matrix.
    """
    roll, yaw, pitch = np.radians([roll, yaw, pitch])
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)

    # Must match OpenCOOD's `transformation_utils.x_to_world` (CARLA /
    # CooperScene left-handed convention). A naive right-handed
    # Rz @ Ry @ Rx flips the sign of the pitch/roll terms in the z-row, which
    # tilts the ground plane and corrupts box z proportional to horizontal
    # distance (3D IoU collapses while BEV looks fine).
    return np.array([
        [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
        [s_p,       -c_p * s_r,                   c_p * c_r],
    ])


# Coordinate conversion: CARLA lidar frame (X-forward, Y-right, Z-up)
# to CV camera convention (X-right, Y-down, Z-forward).
# This is needed because BEVFusion's DepthLSSTransform expects
# lidar2cam in CV convention where Z is the depth axis.
CARLA_TO_CV_CAM = np.array([
    [0, 1, 0, 0],   # cam_X = lidar_Y (right)
    [0, 0, -1, 0],  # cam_Y = -lidar_Z (down)
    [1, 0, 0, 0],   # cam_Z = lidar_X (forward = depth)
    [0, 0, 0, 1]
], dtype=np.float64)


def transform_world_to_ego(
    bbox_world: List[float],
    ego_pose: List[float]
) -> List[float]:
    """Transform bounding box from world coordinates to ego-lidar coordinates.

    Args:
        bbox_world: [x, y, z, dx, dy, dz, yaw] in world frame.
        ego_pose: [x, y, z, roll, yaw, pitch] of ego lidar in world frame.

    Returns:
        [x, y, z, dx, dy, dz, yaw] in ego-lidar frame.
    """
    ego_x, ego_y, ego_z = ego_pose[0], ego_pose[1], ego_pose[2]
    ego_roll, ego_yaw, ego_pitch = ego_pose[3], ego_pose[4], ego_pose[5]

    # Get ego rotation matrix (world to ego)
    R_ego = get_rotation_matrix(ego_roll, ego_yaw, ego_pitch)
    R_ego_inv = R_ego.T  # Inverse rotation

    # Transform position
    pos_world = np.array(bbox_world[:3])
    pos_ego = R_ego_inv @ (pos_world - np.array([ego_x, ego_y, ego_z]))

    # Size stays the same
    size = bbox_world[3:6]

    # Transform yaw (relative to ego)
    yaw_world = bbox_world[6]
    yaw_ego = yaw_world - np.radians(ego_yaw)
    # Normalize to [-pi, pi]
    while yaw_ego > np.pi:
        yaw_ego -= 2 * np.pi
    while yaw_ego < -np.pi:
        yaw_ego += 2 * np.pi

    return [
        float(pos_ego[0]), float(pos_ego[1]), float(pos_ego[2]),
        float(size[0]), float(size[1]), float(size[2]),
        float(yaw_ego)
    ]


def parse_cooperscene_vehicle(
    veh_data: Dict,
    ego_pose: List[float]
) -> Optional[List[float]]:
    """Convert CooperScene vehicle annotation to mmdet3d format.

    CooperScene format:
        location: [x, y, z] - actor origin in world (near ground)
        center: [cx, cy, cz] - bbox center offset in actor's local frame
        extent: [half_l, half_w, half_h] - half sizes
        angle: [roll, yaw, pitch] - in degrees

    mmdet3d format:
        bbox_3d: [x, y, z, dx, dy, dz, yaw] - in ego-lidar coords

    Args:
        veh_data: Vehicle annotation dict from yaml.
        ego_pose: Ego lidar pose [x, y, z, roll, yaw, pitch].

    Returns:
        Bounding box in ego coordinates or None if invalid.
    """
    try:
        # World coordinates
        loc = veh_data['location']  # [x, y, z] actor origin (near ground)
        extent = veh_data['extent']  # [half_l, half_w, half_h]
        angle = veh_data['angle']    # [roll, yaw, pitch] in degrees

        # bbox center offset in actor's local frame
        # center.z is typically ~0.7m (height from ground to bbox center)
        center = veh_data.get('center', [0, 0, 0])

        # Full size (mmdet3d uses full size, not half)
        dx = extent[0] * 2  # length
        dy = extent[1] * 2  # width
        dz = extent[2] * 2  # height

        # Rotate center offset by vehicle's orientation to get world offset
        R_veh = get_rotation_matrix(angle[0], angle[1], angle[2])
        center_world = R_veh @ np.array(center)

        # Actual bbox center = actor location + rotated center offset
        bbox_x = loc[0] + center_world[0]
        bbox_y = loc[1] + center_world[1]
        bbox_z = loc[2] + center_world[2]

        # Yaw in radians
        yaw_world = np.radians(angle[1])

        # Create bbox in world frame
        bbox_world = [bbox_x, bbox_y, bbox_z, dx, dy, dz, yaw_world]

        # Transform to ego-lidar frame
        bbox_ego = transform_world_to_ego(bbox_world, ego_pose)

        return bbox_ego
    except (KeyError, IndexError, TypeError) as e:
        print(f'Warning: Failed to parse vehicle: {e}')
        return None


def convert_pcd_to_bin(pcd_path: str, bin_path: str) -> bool:
    """Convert CooperScene .pcd to .bin format.

    Args:
        pcd_path: Path to input .pcd file.
        bin_path: Path to output .bin file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            'Please install open3d: pip install open3d')

    try:
        # CooperScene .pcd stores the real LiDAR intensity in a dedicated
        # `intensity` field. The legacy `o3d.io.read_point_cloud` only exposes
        # `points`/`colors`, so it silently drops intensity (→ constant 1.0).
        # Use the tensor API and the SAME normalization as OpenCOOD's
        # `pcd_utils.pcd_to_np(isIntensity=True)` (intensity / 65535) so the
        # written .bin is byte-identical to what the cooperative
        # (LoadCooperativePointCloud / OpenCOOD) path produces from the .pcd.
        # The .bin is purely a faster, mmdet3d-native (np.fromfile) container
        # for the same [x, y, z, intensity] data — no information is lost.
        pcd = o3d.t.io.read_point_cloud(pcd_path)
        xyz = pcd.point['positions'].numpy().astype(np.float32)  # (N, 3)
        intensity = (
            pcd.point['intensity'].numpy().astype(np.float32) / 65535.0
        ).reshape(-1, 1)  # (N, 1)

        # Combine: [x, y, z, intensity]
        points = np.hstack([xyz, intensity]).astype(np.float32)  # (N, 4)

        # Save as binary
        points.tofile(bin_path)
        return True
    except Exception as e:
        print(f'Error converting {pcd_path}: {e}')
        return False


def process_single_yaml(
    yaml_path: str,
    data_root: str,
    split: str,
    scenario: str,
    agent_id: str,
    timestamp: str,
    sample_idx: int,
    use_bin: bool = True,
    bin_dir: Optional[str] = None
) -> Optional[Dict]:
    """Process a single yaml annotation file.

    Args:
        yaml_path: Path to yaml file.
        data_root: Root path of the dataset.
        split: Dataset split (train/validate/test).
        scenario: Scenario name.
        agent_id: Agent ID.
        timestamp: Timestamp string.
        sample_idx: Sample index.
        use_bin: Whether to use .bin files (True) or .pcd files (False).

    Returns:
        Info dict or None if failed.
    """
    import yaml

    try:
        with open(yaml_path, 'r') as f:
            meta = yaml.unsafe_load(f)
    except Exception as e:
        print(f'Error loading {yaml_path}: {e}')
        return None

    # Determine point cloud path. Prefer an external --bin-dir (e.g. the
    # shared cooperative .bin tree, real-intensity) so we never re-convert;
    # fall back to a .bin next to the .pcd, then the .pcd itself.
    lidar_path = None
    if bin_dir is not None:
        ext_bin = osp.join(bin_dir, split, scenario, agent_id,
                           f'{timestamp}.bin')
        if osp.exists(ext_bin):
            lidar_path = ext_bin  # absolute path into the shared bin tree
    if lidar_path is None:
        bin_path = osp.join(data_root, split, scenario, agent_id,
                            f'{timestamp}.bin')
        pts_filename = f'{timestamp}.bin' if osp.exists(bin_path) \
            else f'{timestamp}.pcd'
        pts_path = osp.join(data_root, split, scenario, agent_id, pts_filename)
        if not osp.exists(pts_path):
            return None
        lidar_path = osp.join(split, scenario, agent_id, pts_filename)

    # Ego lidar pose: [x, y, z, roll, yaw, pitch]
    ego_pose = meta.get('lidar_pose', meta.get('true_ego_pos'))
    if ego_pose is None:
        print(f'Warning: No ego pose found in {yaml_path}')
        return None

    # Build info dict
    info = {
        'sample_idx': sample_idx,
        'lidar_points': {
            'lidar_path': lidar_path,
            'num_pts_feats': 4,
        },
        'timestamp': float(timestamp),
        'scenario': scenario,
        'agent_id': agent_id,
        'ego_pose': ego_pose,
        'instances': [],
    }

    # Parse camera information
    images = {}
    for cam_id in ['camera0', 'camera1', 'camera2', 'camera3']:
        if cam_id not in meta:
            continue
        cam_data = meta[cam_id]

        # Check if camera image exists
        cam_img_filename = f'{timestamp}_{cam_id}.png'
        cam_img_path = osp.join(
            data_root, split, scenario, agent_id, cam_img_filename)
        if not osp.exists(cam_img_path):
            continue

        # CooperScene extrinsic is lidar2cam in CARLA convention
        # Convert to CV camera convention for BEVFusion
        extrinsic_carla = np.array(
            cam_data['extrinsic'], dtype=np.float64)
        lidar2cam = (CARLA_TO_CV_CAM @ extrinsic_carla).astype(
            np.float32)

        # Intrinsic is standard 3x3 pinhole matrix, no conversion needed
        cam2img = np.array(
            cam_data['intrinsic'], dtype=np.float32)

        images[cam_id] = {
            'img_path': osp.join(
                split, scenario, agent_id, cam_img_filename),
            'lidar2cam': lidar2cam.tolist(),
            'cam2img': cam2img.tolist(),
        }

    if images:
        info['images'] = images

    # Parse vehicle annotations
    vehicles = meta.get('vehicles', {})
    if vehicles:
        for veh_id, veh_data in vehicles.items():
            bbox_ego = parse_cooperscene_vehicle(veh_data, ego_pose)
            if bbox_ego is None:
                continue

            instance = {
                'bbox_3d': bbox_ego,
                'bbox_label_3d': 0,  # vehicle class (only class in CooperScene)
            }

            # Add velocity if available
            if 'speed' in veh_data:
                speed = veh_data['speed'] / 3.6  # km/h to m/s
                yaw = bbox_ego[6]
                vx = speed * np.cos(yaw)
                vy = speed * np.sin(yaw)
                instance['velocity'] = [float(vx), float(vy)]

            info['instances'].append(instance)

    return info


def create_cooperscene_infos(
    data_root: str,
    out_dir: str,
    splits: List[str] = ['train', 'validate', 'test'],
    convert_pcd: bool = False,
    num_workers: int = 8,
    bin_dir: Optional[str] = None
) -> None:
    """Create mmdet3d compatible info files for CooperScene dataset.

    Args:
        data_root: Root path of CooperScene dataset.
        out_dir: Output directory for pkl files.
        splits: List of splits to process.
        convert_pcd: Whether to convert .pcd to .bin files.
        num_workers: Number of workers for parallel processing.
    """
    import yaml

    # Mapping from folder name to output name
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

        print(f'\nProcessing {split}...')

        # Collect all yaml files
        yaml_files = []
        scenarios = [d for d in os.listdir(split_path)
                    if osp.isdir(osp.join(split_path, d))]

        for scenario in scenarios:
            scenario_path = osp.join(split_path, scenario)

            # Get all agents
            agents = [d for d in os.listdir(scenario_path)
                     if osp.isdir(osp.join(scenario_path, d))]

            for agent_id in agents:
                # Only vehicles 1/2/3 act as single-agent ego. Agent 0 is the
                # lidar-only RSU (no camera), which would break the lidar+cam
                # stage; this also matches the cooperative converter.
                if agent_id not in ('1', '2', '3'):
                    continue
                agent_path = osp.join(scenario_path, agent_id)

                # Get all yaml files (annotations)
                # Skip *_additional.yaml and data_protocol.yaml files
                for f in os.listdir(agent_path):
                    if (f.endswith('.yaml') and
                        not f.startswith('data_protocol') and
                        not f.endswith('_additional.yaml')):
                        timestamp = f.replace('.yaml', '')
                        yaml_path = osp.join(agent_path, f)
                        yaml_files.append((yaml_path, scenario, agent_id, timestamp))

        print(f'Found {len(yaml_files)} annotation files')

        # Convert PCD to BIN if requested. Skipped when --bin-dir is set: the
        # .bin already live in that shared tree, so we just reference them.
        if convert_pcd and bin_dir is None:
            print('Converting PCD to BIN...')
            pcd_files = []
            for yaml_path, scenario, agent_id, timestamp in yaml_files:
                agent_path = osp.dirname(yaml_path)
                pcd_path = osp.join(agent_path, f'{timestamp}.pcd')
                bin_path = osp.join(agent_path, f'{timestamp}.bin')
                if osp.exists(pcd_path) and not osp.exists(bin_path):
                    pcd_files.append((pcd_path, bin_path))

            if pcd_files:
                with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    list(tqdm(
                        executor.map(lambda x: convert_pcd_to_bin(x[0], x[1]), pcd_files),
                        total=len(pcd_files),
                        desc='Converting'
                    ))

        # Process yaml files
        infos = []
        for idx, (yaml_path, scenario, agent_id, timestamp) in enumerate(
            tqdm(yaml_files, desc='Processing annotations')
        ):
            info = process_single_yaml(
                yaml_path=yaml_path,
                data_root=data_root,
                split=split,
                scenario=scenario,
                agent_id=agent_id,
                timestamp=timestamp,
                sample_idx=idx,
                use_bin=convert_pcd,
                bin_dir=bin_dir
            )
            if info is not None:
                infos.append(info)

        print(f'Processed {len(infos)} valid samples')

        # Save pkl file
        out_split = split_mapping.get(split, split)
        output = {
            'data_list': infos,
            'metainfo': {
                'classes': ('vehicle',),
                'categories': {'vehicle': 0},  # Required by KittiMetric
                'dataset': 'CooperScene',
                'info_version': '1.0',
            }
        }
        out_path = osp.join(out_dir, f'cooperscene_infos_{out_split}.pkl')
        mmengine.dump(output, out_path)
        print(f'Saved {len(infos)} samples to {out_path}')


def create_cooperscene_dbinfos(
    data_root: str,
    info_path: str,
    out_path: str
) -> None:
    """Create database infos for CooperScene (for ObjectSample augmentation).

    Args:
        data_root: Root path of CooperScene dataset.
        info_path: Path to the info pkl file.
        out_path: Output path for dbinfos pkl file.
    """
    print(f'Creating dbinfos from {info_path}...')

    data = mmengine.load(info_path)
    infos = data['data_list']

    # Database for each class
    db_infos = {'vehicle': []}

    for info in tqdm(infos, desc='Processing'):
        pts_path = osp.join(data_root, info['lidar_points']['lidar_path'])

        # Load points
        if pts_path.endswith('.bin'):
            points = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
        else:
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(pts_path)
                points = np.asarray(pcd.points, dtype=np.float32)
                if points.shape[1] == 3:
                    intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                    points = np.hstack([points, intensity])
            except Exception:
                continue

        for idx, instance in enumerate(info['instances']):
            bbox = instance['bbox_3d']

            # Count points in box (simplified)
            # For accurate count, you'd need proper box intersection
            x, y, z, dx, dy, dz, yaw = bbox
            mask = (
                (np.abs(points[:, 0] - x) < dx / 2) &
                (np.abs(points[:, 1] - y) < dy / 2) &
                (np.abs(points[:, 2] - z) < dz / 2)
            )
            num_pts = mask.sum()

            if num_pts < 5:
                continue

            db_info = {
                'name': 'vehicle',
                'path': info['lidar_points']['lidar_path'],
                'box3d_lidar': bbox,
                'num_points_in_gt': int(num_pts),
                'sample_idx': info['sample_idx'],
            }
            db_infos['vehicle'].append(db_info)

    # Save
    mmengine.dump(db_infos, out_path)
    print(f'Saved {len(db_infos["vehicle"])} vehicle instances to {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='CooperScene Dataset Converter')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/cooperscene',
        help='Root path of CooperScene dataset'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/cooperscene',
        help='Output directory for pkl files'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validate', 'test'],
        help='Splits to process'
    )
    parser.add_argument(
        '--convert-pcd',
        action='store_true',
        help='Convert .pcd files to .bin files'
    )
    parser.add_argument(
        '--bin-dir',
        default=None,
        help='Reference existing .bin from this shared tree '
             '(mirrors <split>/<scenario>/<agent>/<ts>.bin) instead of '
             'creating .bin next to the .pcd. lidar_path becomes absolute.'
    )
    parser.add_argument(
        '--create-dbinfos',
        action='store_true',
        help='Create database infos for ObjectSample'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of workers for parallel processing'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Create info files
    create_cooperscene_infos(
        data_root=args.data_root,
        out_dir=args.out_dir,
        splits=args.splits,
        convert_pcd=args.convert_pcd,
        num_workers=args.num_workers,
        bin_dir=args.bin_dir
    )

    # Create dbinfos if requested
    if args.create_dbinfos:
        train_info_path = osp.join(args.out_dir, 'cooperscene_infos_train.pkl')
        if osp.exists(train_info_path):
            dbinfo_path = osp.join(args.out_dir, 'cooperscene_dbinfos_train.pkl')
            create_cooperscene_dbinfos(
                data_root=args.data_root,
                info_path=train_info_path,
                out_path=dbinfo_path
            )
