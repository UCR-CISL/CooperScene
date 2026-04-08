#!/usr/bin/env python3
"""CooperScene camera calibration tuning.

For each agent, performs a grid search over pitch/yaw/roll deltas and
focal scale to find the best lidar-to-camera extrinsic adjustment.
Scores by projecting lidar points onto the image and measuring edge
alignment (projected lidar points on image edges = good calibration).

Saves 2 proof images per (scene, agent):
  1. Lidar points colored by depth overlaid on camera image
  2. 3D bounding boxes projected onto camera image

Usage:
    python tools/dataset_converters/cooperscene_calibration.py \
        --data-root /path/to/250928 \
        --out-dir /path/to/output
"""

import argparse
import json
import os
from itertools import product
from os import path as osp

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

# ── Base calibration from SensorPlatform ──
CALIBRATION = {
    't1': {
        'intrinsic': [
            [1810.6983377214349, 0.0, 970.0037900558976],
            [0.0, 1808.4746114412712, 526.7243033929532],
            [0.0, 0.0, 1.0]],
        'T_lidar_to_cam': [
            [1.97675856e-02, -9.98998871e-01, -4.01310140e-02, -1.14236290e-04],
            [-4.47204289e-02, 3.92152089e-02, -9.98229558e-01, -1.75248879e-01],
            [9.98803948e-01, 2.15272644e-02, -4.39004680e-02, -5.25987245e-02],
            [0.0, 0.0, 0.0, 1.0]]
    },
    't2': {
        'intrinsic': [
            [1807.294188309111, 0.0, 959.1274184107824],
            [0.0, 1805.1344921703185, 562.0235332672737],
            [0.0, 0.0, 1.0]],
        'T_lidar_to_cam': [
            [0.00383388, -0.99998521, -0.00385735, -0.0013977],
            [-0.00271154, 0.00384697, -0.99998892, -0.23015363],
            [0.99998897, 0.0038443, -0.00269675, -0.07767752],
            [0.0, 0.0, 0.0, 1.0]]
    },
    't3': {
        'intrinsic': [
            [1814.1641900346326, 0.0, 951.718698093622],
            [0.0, 1812.1352460466578, 568.1954332460546],
            [0.0, 0.0, 1.0]],
        'T_lidar_to_cam': [
            [0.064023, -0.99792386, -0.00700149, -0.00332843],
            [-0.01444825, 0.00608825, -0.99987708, -0.26612881],
            [0.99784383, 0.06411629, -0.01402846, -0.0864176],
            [0.0, 0.0, 0.0, 1.0]]
    }
}

AGENT_CALIB_MAP = {'1': 't1', '2': 't2', '3': 't3'}


# ── Geometry helpers ──

def get_rotation_matrix(roll, yaw, pitch):
    roll, yaw, pitch = np.radians([roll, yaw, pitch])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_adjustment(T_base, K_base, pitch_deg, yaw_deg, roll_deg,
                     focal_scale):
    """Apply pitch/yaw/roll delta to extrinsic and focal scale to intrinsic."""
    T = T_base.copy()
    for axis, deg in [('x', pitch_deg), ('y', yaw_deg), ('z', roll_deg)]:
        if abs(deg) > 1e-8:
            R_delta = Rot.from_euler(axis, deg, degrees=True).as_matrix()
            T[:3, :3] = R_delta @ T[:3, :3]
            T[:3, 3] = R_delta @ T[:3, 3]

    K = K_base.copy()
    K[0, 0] *= focal_scale
    K[1, 1] *= focal_scale
    return T, K


def load_points_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def load_points_pcd(pcd_path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[1] == 3:
        pts = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    return pts


def load_points(data_root, scenario, agent_id, timestamp):
    bin_path = osp.join(data_root, scenario, agent_id, f'{timestamp}.bin')
    if osp.exists(bin_path):
        return load_points_bin(bin_path)
    pcd_path = osp.join(data_root, scenario, agent_id, f'{timestamp}.pcd')
    return load_points_pcd(pcd_path)


def project_points(pts_lidar, T_l2c, K, img_hw):
    """Project lidar xyz to image pixels. Returns (uv_int, depth, mask)."""
    h, w = img_hw
    pts_h = np.hstack([pts_lidar[:, :3],
                       np.ones((len(pts_lidar), 1), dtype=np.float64)])
    pts_cam = (T_l2c @ pts_h.T).T[:, :3]

    z = pts_cam[:, 2]
    valid = z > 0.5

    uv = np.full((len(pts_lidar), 2), -1, dtype=np.float64)
    if valid.sum() > 0:
        uvw = (K @ pts_cam[valid].T).T
        uv_v = uvw[:, :2] / uvw[:, 2:3]
        inb = ((uv_v[:, 0] >= 0) & (uv_v[:, 0] < w) &
               (uv_v[:, 1] >= 0) & (uv_v[:, 1] < h))
        idx = np.where(valid)[0]
        valid[idx[~inb]] = False
        uv[idx[inb]] = uv_v[inb]

    return uv[valid].astype(np.int32), z[valid], valid


def compute_edge_score(pts_lidar, T_l2c, K, img_gray):
    """Score: mean image gradient magnitude at projected lidar points."""
    h, w = img_gray.shape
    # Sobel edge magnitude
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad = (grad / (grad.max() + 1e-8) * 255).astype(np.uint8)

    uv, depth, mask = project_points(pts_lidar, T_l2c, K, (h, w))
    if len(uv) < 50:
        return -1.0

    vals = grad[uv[:, 1], uv[:, 0]].astype(np.float64)
    return float(np.mean(vals))


# ── Bounding box helpers ──

def bbox_corners_lidar(center_xyz, quat_wxyz, size_lwh):
    """8 corners from lidar-frame annotation with quaternion.

    Args:
        center_xyz: (3,) bbox center in lidar coords.
        quat_wxyz: (4,) quaternion [qw, qx, qy, qz].
        size_lwh: (3,) [length, width, height].

    Returns:
        (8, 3) corners in lidar coords.
    """
    l, w, h = size_lwh
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners_local = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz],
    ], dtype=np.float64)
    qw, qx, qy, qz = quat_wxyz
    rot = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    return corners_local @ rot.T + np.array(center_xyz, dtype=np.float64)


def draw_bbox_on_image(img, corners_cam, K, color=(0, 255, 0), thickness=2):
    """Draw 3D bbox wireframe on image from camera-frame corners."""
    z = corners_cam[:, 2]
    if np.any(z <= 0.1):
        return img
    uvw = (K @ corners_cam.T).T
    uv = (uvw[:, :2] / uvw[:, 2:3]).astype(np.int32)
    h, w = img.shape[:2]
    if np.any((uv[:, 0] < -500) | (uv[:, 0] > w + 500) |
              (uv[:, 1] < -500) | (uv[:, 1] > h + 500)):
        return img
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, thickness,
                 lineType=cv2.LINE_AA)
    return img


# ── Visualization ──

def draw_lidar_overlay(img, pts_lidar, T_l2c, K):
    """Overlay lidar points colored by depth on camera image."""
    out = img.copy()
    h, w = img.shape[:2]
    uv, depth, mask = project_points(pts_lidar, T_l2c, K, (h, w))
    if len(uv) == 0:
        return out

    d_min, d_max = np.percentile(depth, [2, 98])
    d_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
    colors = cv2.applyColorMap(
        (d_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET).reshape(-1, 3)

    for (u, v), c in zip(uv, colors):
        cv2.circle(out, (u, v), 1, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def load_lidar_labels(labels_root, scenario, agent_id, timestamp):
    """Load lidar-frame labels from release labels_refined_separate.

    Args:
        labels_root: Root of release data (e.g., .../release/250928).
        scenario: Scenario ID.
        agent_id: Agent ID.
        timestamp: Frame timestamp.

    Returns:
        List of dicts with 'center', 'quat_wxyz', 'size_lwh', 'obj_id'.
    """
    label_path = osp.join(
        labels_root, scenario, agent_id,
        'labels_refined_separate', f'{timestamp}.yaml')
    if not osp.exists(label_path):
        return []

    with open(label_path, 'r') as f:
        data = yaml.safe_load(f)

    objects = []
    for obj_id, obj in data.get('objects', {}).items():
        arr = obj.get('bbox_lidar_coords')
        if arr is None:
            continue
        arr = np.array(arr, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 10:
            continue
        objects.append({
            'obj_id': obj_id,
            'center': arr[0:3],
            'quat_wxyz': arr[3:7],
            'size_lwh': arr[7:10],
        })
    return objects


def draw_bbox_overlay(img, lidar_labels, T_l2c, K):
    """Project lidar-frame bboxes onto image.

    Args:
        img: BGR image.
        lidar_labels: List from load_lidar_labels().
        T_l2c: 4x4 lidar-to-camera extrinsic.
        K: 3x3 camera intrinsic.
    """
    out = img.copy()

    for obj in lidar_labels:
        corners_l = bbox_corners_lidar(
            obj['center'], obj['quat_wxyz'], obj['size_lwh'])
        # Lidar -> camera
        corners_h = np.hstack([corners_l, np.ones((8, 1), dtype=np.float64)])
        corners_c = (T_l2c @ corners_h.T).T[:, :3]
        out = draw_bbox_on_image(out, corners_c, K, color=(0, 255, 0),
                                 thickness=2)
        # Draw object ID
        center_c = (T_l2c @ np.append(obj['center'], 1.0))[:3]
        if center_c[2] > 0.5:
            uvw = K @ center_c
            u, v = int(uvw[0] / uvw[2]), int(uvw[1] / uvw[2])
            h, w = out.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                cv2.putText(out, str(obj['obj_id']), (u, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Main calibration pipeline ──

def find_best_calibration(pts_list, img_gray_list, T_base, K_base):
    """Grid search for best pitch/yaw/roll/focal adjustment.

    Two-stage: coarse then fine.
    """
    # Coarse search
    pitches = np.arange(-5, 4, 2.0)
    yaws = np.arange(-5, 4, 2.0)
    rolls = np.arange(-3, 4, 2.0)
    focals = [0.90, 0.95, 1.0]

    best_score = -1e9
    best_params = (0, 0, 0, 1.0)

    combos = list(product(pitches, yaws, rolls, focals))
    for p, y, r, f in tqdm(combos, desc='  Coarse search', leave=False):
        T_adj, K_adj = apply_adjustment(T_base, K_base, p, y, r, f)
        score = 0
        for pts, gray in zip(pts_list, img_gray_list):
            score += compute_edge_score(pts, T_adj, K_adj, gray)
        score /= len(pts_list)
        if score > best_score:
            best_score = score
            best_params = (p, y, r, f)

    # Fine search around best
    bp, by, br, bf = best_params
    pitches_f = np.arange(bp - 1.5, bp + 2.0, 0.5)
    yaws_f = np.arange(by - 1.5, by + 2.0, 0.5)
    rolls_f = np.arange(br - 1.5, br + 2.0, 0.5)
    focals_f = [max(0.85, bf - 0.05), bf, min(1.15, bf + 0.05)]

    combos_f = list(product(pitches_f, yaws_f, rolls_f, focals_f))
    for p, y, r, f in tqdm(combos_f, desc='  Fine search', leave=False):
        T_adj, K_adj = apply_adjustment(T_base, K_base, p, y, r, f)
        score = 0
        for pts, gray in zip(pts_list, img_gray_list):
            score += compute_edge_score(pts, T_adj, K_adj, gray)
        score /= len(pts_list)
        if score > best_score:
            best_score = score
            best_params = (p, y, r, f)

    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(
        description='CooperScene camera calibration tuning')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root of benchmark dataset (e.g., .../250928)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory for proof images and params')
    parser.add_argument('--labels-root', type=str, default=None,
                        help='Root of release data with labels_refined_separate')
    parser.add_argument('--num-frames', type=int, default=5,
                        help='Number of frames to sample per agent for tuning')
    args = parser.parse_args()

    data_root = args.data_root
    out_dir = args.out_dir or osp.join(data_root, 'calibration_results')
    os.makedirs(out_dir, exist_ok=True)

    # Discover scenarios and agents
    scenarios = sorted([
        d for d in os.listdir(data_root)
        if osp.isdir(osp.join(data_root, d)) and not d.endswith('_infra')
        and not d.endswith('.pkl')
    ])

    all_results = {}

    for scenario in scenarios:
        scenario_path = osp.join(data_root, scenario)
        agents = sorted([
            d for d in os.listdir(scenario_path)
            if osp.isdir(osp.join(scenario_path, d))
        ])

        for agent_id in agents:
            calib_key = AGENT_CALIB_MAP.get(agent_id)
            if calib_key is None:
                print(f'Skip agent {agent_id}: no calibration data')
                continue

            print(f'\n=== Scenario {scenario}, Agent {agent_id} ({calib_key}) ===')

            agent_path = osp.join(scenario_path, agent_id)

            # Get frame list
            frames = sorted([
                f.replace('.yaml', '') for f in os.listdir(agent_path)
                if f.endswith('.yaml')
            ])
            if not frames:
                continue

            # Sample frames evenly
            n = min(args.num_frames, len(frames))
            indices = np.linspace(0, len(frames) - 1, n, dtype=int)
            sample_frames = [frames[i] for i in indices]

            # Base calibration
            T_base = np.array(CALIBRATION[calib_key]['T_lidar_to_cam'],
                              dtype=np.float64)
            K_base = np.array(CALIBRATION[calib_key]['intrinsic'],
                              dtype=np.float64)

            # Load data for sampled frames
            pts_list = []
            img_gray_list = []
            img_color_list = []
            meta_list = []
            frame_ids = []

            for ts in sample_frames:
                pts = load_points(data_root, scenario, agent_id, ts)
                img_path = osp.join(agent_path, f'{ts}_camera0.png')
                if not osp.exists(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue

                yaml_path = osp.join(agent_path, f'{ts}.yaml')
                with open(yaml_path, 'r') as f:
                    meta = yaml.unsafe_load(f)

                pts_list.append(pts)
                img_gray_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                img_color_list.append(img)
                meta_list.append(meta)
                frame_ids.append(ts)

            if not pts_list:
                print('  No valid frames found, skip.')
                continue

            print(f'  Using {len(pts_list)} frames: {frame_ids}')

            # Grid search
            best_params, best_score = find_best_calibration(
                pts_list, img_gray_list, T_base, K_base)

            pitch, yaw, roll, focal = best_params
            print(f'  Best: pitch={pitch:.1f}, yaw={yaw:.1f}, '
                  f'roll={roll:.1f}, focal={focal:.2f}  '
                  f'(score={best_score:.2f})')

            # Compute final adjusted calibration
            T_final, K_final = apply_adjustment(
                T_base, K_base, pitch, yaw, roll, focal)

            # Save 2 proof images (first and last sampled frame)
            proof_indices = [0, len(pts_list) - 1]
            for pi in proof_indices:
                ts = frame_ids[pi]
                img = img_color_list[pi]
                pts = pts_list[pi]

                # Image 1: lidar projection
                img_lidar = draw_lidar_overlay(img, pts, T_final, K_final)
                cv2.putText(img_lidar,
                            f'Agent {agent_id} | pitch={pitch:.1f} '
                            f'yaw={yaw:.1f} roll={roll:.1f} f={focal:.2f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                fname1 = f'scene{scenario}_agent{agent_id}_{ts}_lidar.png'
                cv2.imwrite(osp.join(out_dir, fname1), img_lidar)

                # Image 2: bbox projection using lidar-frame labels
                lidar_labels = []
                if args.labels_root:
                    lidar_labels = load_lidar_labels(
                        args.labels_root, scenario, agent_id, ts)
                if lidar_labels:
                    img_bbox = draw_bbox_overlay(
                        img, lidar_labels, T_final, K_final)
                else:
                    # Fallback: just show lidar overlay with note
                    img_bbox = img_lidar.copy()
                    cv2.putText(img_bbox, 'No lidar-frame labels',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                cv2.putText(img_bbox,
                            f'Agent {agent_id} | pitch={pitch:.1f} '
                            f'yaw={yaw:.1f} roll={roll:.1f} f={focal:.2f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                fname2 = f'scene{scenario}_agent{agent_id}_{ts}_bbox.png'
                cv2.imwrite(osp.join(out_dir, fname2), img_bbox)

                print(f'  Saved: {fname1}')
                print(f'  Saved: {fname2}')

            # Store results
            all_results[f'scene{scenario}_agent{agent_id}'] = {
                'calib_key': calib_key,
                'pitch_delta': float(pitch),
                'yaw_delta': float(yaw),
                'roll_delta': float(roll),
                'focal_scale': float(focal),
                'score': float(best_score),
                'T_lidar_to_cam': T_final.tolist(),
                'intrinsic': K_final.tolist(),
            }

    # Save all calibration results
    result_path = osp.join(out_dir, 'calibration_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nCalibration results saved to {result_path}')

    # Print summary
    print('\n=== Summary ===')
    for key, val in all_results.items():
        print(f'{key}: pitch={val["pitch_delta"]:.1f}, '
              f'yaw={val["yaw_delta"]:.1f}, '
              f'roll={val["roll_delta"]:.1f}, '
              f'focal={val["focal_scale"]:.2f} '
              f'(score={val["score"]:.2f})')


if __name__ == '__main__':
    main()
