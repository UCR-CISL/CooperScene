"""Pre-compute trajectory GT for OPV2V motion prediction.

This script post-processes existing OPV2V info pkl files to add:
- Future trajectories (12 steps at 2Hz = 6 seconds)
- Past trajectories (4 steps at 2Hz = 2 seconds)
- Trajectory validity masks
- Vehicle tracking IDs (CARLA actor IDs)
- Temporal sequence indices (prev/next frame pointers)

The trajectory GT is computed by:
1. Parsing yaml files to get vehicle IDs and world positions
2. Matching vehicle IDs across frames (skipping by step_stride)
3. Computing positions relative to the current ego frame

OPV2V records at 10Hz (timestamps increment by ~0.1s). To produce 2Hz
waypoints that match UniAD's convention, use --step-stride 5 (default)
so that each trajectory step spans 5 raw frames = 0.5s. With 12 future
steps this covers 6 seconds; with 4 past steps, 2 seconds.

Usage:
    python tools/dataset_converters/opv2v_trajectory_converter.py \
        --info-path /home/bowu/cluster/OPV2V/opv2v_coop_infos_train.pkl \
        --data-root /home/bowu/cluster/OPV2V/ \
        --out-path /home/bowu/cluster/OPV2V/opv2v_coop_motion_infos_train.pkl \
        --future-steps 12 --past-steps 4 --step-stride 5
"""

import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import numpy as np
import yaml
from tqdm import tqdm


def get_rotation_matrix(roll, yaw, pitch):
    """Get rotation matrix from Euler angles (in degrees)."""
    roll, yaw, pitch = np.radians([roll, yaw, pitch])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def pose_to_matrix(pose):
    """Convert [x, y, z, roll, yaw, pitch] to 4x4 transform matrix."""
    R = get_rotation_matrix(pose[3], pose[4], pose[5])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T


def get_vehicle_world_positions(yaml_path, ego_pose):
    """Parse yaml to get vehicle world positions and bbox in ego frame.

    Args:
        yaml_path: Path to annotation yaml.
        ego_pose: [x, y, z, roll, yaw, pitch] of ego.

    Returns:
        dict: {vehicle_id: {'world_pos': (3,), 'ego_pos': (3,)}}
    """
    try:
        with open(yaml_path, 'r') as f:
            meta = yaml.unsafe_load(f)
    except Exception:
        return {}

    vehicles = meta.get('vehicles', {})
    if not vehicles:
        return {}

    R_ego = get_rotation_matrix(ego_pose[3], ego_pose[4], ego_pose[5])
    R_ego_inv = R_ego.T
    ego_pos = np.array(ego_pose[:3])

    result = {}
    for veh_id, veh_data in vehicles.items():
        try:
            loc = veh_data['location']
            center = veh_data.get('center', [0, 0, 0])
            angle = veh_data['angle']

            # Rotate center offset by vehicle's orientation
            R_veh = get_rotation_matrix(angle[0], angle[1], angle[2])
            center_world = R_veh @ np.array(center)

            # World position of bbox center
            world_pos = np.array([
                loc[0] + center_world[0],
                loc[1] + center_world[1],
                loc[2] + center_world[2],
            ])

            # Ego-frame position
            ego_frame_pos = R_ego_inv @ (world_pos - ego_pos)

            result[int(veh_id)] = {
                'world_pos': world_pos,
                'ego_pos': ego_frame_pos,
            }
        except (KeyError, IndexError, TypeError):
            continue

    return result


def build_temporal_index(infos):
    """Build temporal sequence index.

    Groups frames by (scenario, agent_id) and sorts by timestamp.

    Returns:
        dict: (scenario, agent_id) -> sorted list of (timestamp, info_index)
    """
    scene_agent_frames = defaultdict(list)
    for idx, info in enumerate(infos):
        key = (info['scenario'], str(info['agent_id']))
        scene_agent_frames[key].append((info['timestamp'], idx))

    for key in scene_agent_frames:
        scene_agent_frames[key].sort(key=lambda x: x[0])

    return scene_agent_frames


def add_temporal_pointers(infos, scene_agent_frames):
    """Add prev/next frame index pointers."""
    for key, frames in scene_agent_frames.items():
        for i, (ts, idx) in enumerate(frames):
            infos[idx]['prev_idx'] = frames[i - 1][1] if i > 0 else -1
            infos[idx]['next_idx'] = (
                frames[i + 1][1] if i < len(frames) - 1 else -1)
            infos[idx]['sequence_idx'] = i
            infos[idx]['sequence_length'] = len(frames)


def resolve_yaml_path(info, data_root):
    """Find the yaml file path for a given info dict."""
    scenario = info['scenario']
    agent_id = str(info['agent_id'])
    timestamp = info['timestamp']

    # Extract split from lidar_path
    lidar_path = info['lidar_points']['lidar_path']
    split = lidar_path.split('/')[0]

    # Try different timestamp formats
    ts_strs = [f'{int(timestamp):06d}', str(int(timestamp))]
    for ts_str in ts_strs:
        yaml_path = osp.join(
            data_root, split, scenario, agent_id, f'{ts_str}.yaml')
        if osp.exists(yaml_path):
            return yaml_path

    return None


def compute_trajectories(infos, scene_agent_frames, data_root,
                         future_steps=12, past_steps=4, step_stride=5):
    """Compute trajectory GT for all frames.

    For each frame:
    1. Parse current yaml to get vehicle IDs and positions
    2. Look up future/past frames spaced by ``step_stride`` raw frames
    3. Compute relative displacements in ego frame

    OPV2V is recorded at 10 Hz.  With the default ``step_stride=5`` we
    sample every 5th raw frame, giving an effective 2 Hz trajectory rate
    (0.5 s between waypoints).  12 future steps then cover 6 s; 4 past
    steps cover 2 s -- matching the UniAD convention.

    Args:
        step_stride (int): Number of raw frames to skip between each
            trajectory waypoint.  At 10 Hz, stride=5 -> 2 Hz effective.
    """
    effective_hz = 10.0 / step_stride
    step_dt = step_stride / 10.0
    print(f'Computing trajectories: {past_steps} past, {future_steps} future '
          f'(stride={step_stride}, {effective_hz:.1f}Hz, {step_dt}s/step)...')

    # Step 1: Build vehicle world position cache for all frames
    # This avoids parsing the same yaml multiple times
    print('Step 1/2: Parsing yaml files for vehicle positions...')
    frame_vehicles = {}  # info_idx -> {veh_id: {'world_pos': ..., 'ego_pos': ...}}

    for idx, info in enumerate(tqdm(infos, desc='Parsing yamls')):
        yaml_path = resolve_yaml_path(info, data_root)
        if yaml_path is not None:
            frame_vehicles[idx] = get_vehicle_world_positions(
                yaml_path, info['ego_pose'])
        else:
            frame_vehicles[idx] = {}

    # Step 2: Compute trajectories
    print('Step 2/2: Computing trajectory displacements...')
    total_with_traj = 0

    for key, frames in tqdm(scene_agent_frames.items(),
                            desc='Computing trajectories'):
        n_frames = len(frames)

        for frame_i, (ts_cur, idx_cur) in enumerate(frames):
            info = infos[idx_cur]
            cur_vehicles = frame_vehicles[idx_cur]

            # Get all vehicle IDs in current frame
            veh_ids = sorted(cur_vehicles.keys())

            if not veh_ids:
                info['gt_fut_traj'] = np.zeros(
                    (0, future_steps, 2), dtype=np.float32)
                info['gt_fut_traj_mask'] = np.zeros(
                    (0, future_steps), dtype=np.float32)
                info['gt_past_traj'] = np.zeros(
                    (0, past_steps, 2), dtype=np.float32)
                info['gt_past_traj_mask'] = np.zeros(
                    (0, past_steps), dtype=np.float32)
                info['gt_vehicle_ids'] = []
                continue

            n_veh = len(veh_ids)
            ego_pose = info['ego_pose']
            T_ego = pose_to_matrix(ego_pose)
            T_ego_inv = np.linalg.inv(T_ego)

            fut_traj = np.zeros((n_veh, future_steps, 2), dtype=np.float32)
            fut_mask = np.zeros((n_veh, future_steps), dtype=np.float32)
            past_traj = np.zeros((n_veh, past_steps, 2), dtype=np.float32)
            past_mask = np.zeros((n_veh, past_steps), dtype=np.float32)

            for v_i, vid in enumerate(veh_ids):
                cur_world = cur_vehicles[vid]['world_pos']

                # Transform current position to ego frame for reference
                cur_h = np.append(cur_world, 1.0)
                cur_ego = (T_ego_inv @ cur_h)[:3]

                # Future trajectory
                for step in range(future_steps):
                    future_frame_i = frame_i + (step + 1) * step_stride
                    if future_frame_i >= n_frames:
                        break
                    _, idx_future = frames[future_frame_i]
                    fut_vehicles = frame_vehicles.get(idx_future, {})

                    if vid in fut_vehicles:
                        fut_world = fut_vehicles[vid]['world_pos']
                        fut_h = np.append(fut_world, 1.0)
                        fut_ego = (T_ego_inv @ fut_h)[:3]
                        # Relative displacement from current position
                        fut_traj[v_i, step, 0] = fut_ego[0] - cur_ego[0]
                        fut_traj[v_i, step, 1] = fut_ego[1] - cur_ego[1]
                        fut_mask[v_i, step] = 1.0

                # Past trajectory
                for step in range(past_steps):
                    past_frame_i = frame_i - (step + 1) * step_stride
                    if past_frame_i < 0:
                        break
                    _, idx_past = frames[past_frame_i]
                    past_vehicles = frame_vehicles.get(idx_past, {})

                    if vid in past_vehicles:
                        past_world = past_vehicles[vid]['world_pos']
                        past_h = np.append(past_world, 1.0)
                        past_ego = (T_ego_inv @ past_h)[:3]
                        past_traj[v_i, step, 0] = past_ego[0] - cur_ego[0]
                        past_traj[v_i, step, 1] = past_ego[1] - cur_ego[1]
                        past_mask[v_i, step] = 1.0

            info['gt_fut_traj'] = fut_traj
            info['gt_fut_traj_mask'] = fut_mask
            info['gt_past_traj'] = past_traj
            info['gt_past_traj_mask'] = past_mask
            info['gt_vehicle_ids'] = veh_ids

            if n_veh > 0:
                total_with_traj += 1

    print(f'Computed trajectories for {total_with_traj} frames with vehicles')


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute trajectory GT for OPV2V')
    parser.add_argument(
        '--info-path', type=str, required=True,
        help='Path to existing opv2v_coop_infos_*.pkl')
    parser.add_argument(
        '--data-root', type=str, default='/home/bowu/cluster/OPV2V/',
        help='Root path of OPV2V dataset (for yaml parsing)')
    parser.add_argument(
        '--out-path', type=str, default=None,
        help='Output path. Default: replace _infos_ with _motion_infos_')
    parser.add_argument(
        '--future-steps', type=int, default=12,
        help='Number of future trajectory steps (default 12 = 6s at 2Hz)')
    parser.add_argument(
        '--past-steps', type=int, default=4,
        help='Number of past trajectory steps (default 4 = 2s at 2Hz)')
    parser.add_argument(
        '--step-stride', type=int, default=5,
        help='Number of raw 10Hz frames to skip between trajectory '
             'waypoints. stride=5 gives 2Hz effective rate (default 5)')
    args = parser.parse_args()

    if args.out_path is None:
        base, ext = osp.splitext(args.info_path)
        args.out_path = base.replace('_infos_', '_motion_infos_') + ext
        if args.out_path == args.info_path:
            args.out_path = f'{base}_motion{ext}'

    print(f'Loading {args.info_path}...')
    data = mmengine.load(args.info_path)
    infos = data['data_list']
    print(f'Loaded {len(infos)} samples')

    # Build temporal index
    scene_agent_frames = build_temporal_index(infos)
    print(f'Built temporal index: {len(scene_agent_frames)} sequences')
    for key, frames in list(scene_agent_frames.items())[:3]:
        print(f'  {key}: {len(frames)} frames')

    # Add temporal pointers
    add_temporal_pointers(infos, scene_agent_frames)

    # Compute trajectories
    compute_trajectories(
        infos, scene_agent_frames, args.data_root,
        future_steps=args.future_steps,
        past_steps=args.past_steps,
        step_stride=args.step_stride)

    # Update metainfo
    effective_hz = 10.0 / args.step_stride
    data['metainfo']['motion_info'] = {
        'future_steps': args.future_steps,
        'past_steps': args.past_steps,
        'step_stride': args.step_stride,
        'raw_freq_hz': 10,
        'sample_freq_hz': effective_hz,
    }

    # Save
    print(f'Saving to {args.out_path}...')
    mmengine.dump(data, args.out_path)
    print('Done!')

    # Print stats
    n_with_fut = sum(
        1 for info in infos
        if 'gt_fut_traj' in info and info['gt_fut_traj'].shape[0] > 0)
    n_total_veh = sum(
        info['gt_fut_traj'].shape[0]
        for info in infos if 'gt_fut_traj' in info)
    avg_veh = n_total_veh / max(n_with_fut, 1)

    # Check trajectory validity coverage
    full_future = 0
    partial_future = 0
    for info in infos:
        if 'gt_fut_traj_mask' not in info:
            continue
        mask = info['gt_fut_traj_mask']
        if mask.shape[0] == 0:
            continue
        # Count vehicles with full future trajectory
        full_future += (mask.sum(axis=1) == args.future_steps).sum()
        partial_future += (mask.sum(axis=1) > 0).sum()

    step_dt = args.step_stride / 10.0
    fut_horizon = args.future_steps * step_dt
    past_horizon = args.past_steps * step_dt

    print(f'\nStats:')
    print(f'  Step stride: {args.step_stride} raw frames '
          f'({step_dt}s/step, {effective_hz:.1f}Hz effective)')
    print(f'  Future horizon: {args.future_steps} steps x '
          f'{step_dt}s = {fut_horizon:.1f}s')
    print(f'  Past horizon: {args.past_steps} steps x '
          f'{step_dt}s = {past_horizon:.1f}s')
    print(f'  Frames with trajectory GT: {n_with_fut}/{len(infos)}')
    print(f'  Total vehicle trajectories: {n_total_veh}')
    print(f'  Average vehicles per frame: {avg_veh:.1f}')
    print(f'  Vehicles with full {args.future_steps}-step future: '
          f'{full_future}')
    print(f'  Vehicles with partial future: {partial_future}')


if __name__ == '__main__':
    main()
