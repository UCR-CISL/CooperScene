"""Generate K-Means motion anchors from GT trajectories.

Clusters GT future trajectories into K modes and saves the cluster centers
as anchor templates for MotionHead initialization (fixes mode collapse).

This matches UniAD's approach of using K-Means anchors instead of random
learnable embeddings for the intention queries.

Usage:
    python tools/generate_motion_anchors.py \
        --pkl /home/bowu/data/opv2v_4_mmdet3d_bevfusion/opv2v_motion_infos_train.pkl \
        --num-modes 6 \
        --predict-steps 12 \
        --output projects/UniAD/configs/motion_anchors_k6.npy
"""

import argparse
import pickle
import numpy as np
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser(
        description='Generate K-Means motion anchors from GT trajectories')
    parser.add_argument('--pkl', required=True,
                        help='Path to motion info pkl file')
    parser.add_argument('--num-modes', type=int, default=6,
                        help='Number of trajectory modes (K)')
    parser.add_argument('--predict-steps', type=int, default=12,
                        help='Number of future prediction steps')
    parser.add_argument('--output', required=True,
                        help='Output .npy file for anchor templates')
    parser.add_argument('--min-valid-steps', type=int, default=6,
                        help='Minimum valid future steps to include trajectory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for K-Means')
    args = parser.parse_args()

    # Load pkl
    print(f'Loading {args.pkl}...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    data_list = data.get('data_list', data.get('infos', []))
    print(f'Found {len(data_list)} frames')

    # Collect all valid GT future trajectories
    all_trajs = []
    for info in data_list:
        gt_fut_traj = info.get('gt_fut_traj', None)
        gt_fut_traj_mask = info.get('gt_fut_traj_mask', None)
        if gt_fut_traj is None:
            continue

        gt_fut_traj = np.array(gt_fut_traj)  # (N, T, 2)
        gt_fut_traj_mask = np.array(gt_fut_traj_mask)  # (N, T)

        if gt_fut_traj.ndim != 3:
            continue

        N, T, _ = gt_fut_traj.shape
        T_use = min(T, args.predict_steps)

        for i in range(N):
            mask = gt_fut_traj_mask[i, :T_use]
            valid_steps = int(mask.sum())
            if valid_steps < args.min_valid_steps:
                continue

            traj = gt_fut_traj[i, :args.predict_steps, :2].copy()
            # Zero-pad if shorter than predict_steps
            if T_use < args.predict_steps:
                padded = np.zeros((args.predict_steps, 2))
                padded[:T_use] = traj[:T_use]
                # Extend with last valid position
                if T_use > 0:
                    padded[T_use:] = traj[T_use - 1]
                traj = padded

            all_trajs.append(traj)

    all_trajs = np.array(all_trajs)  # (M, T, 2)
    print(f'Collected {len(all_trajs)} valid trajectories')

    # Flatten for K-Means: (M, T*2)
    M, T, _ = all_trajs.shape
    flat_trajs = all_trajs.reshape(M, -1)

    # Run K-Means
    print(f'Running K-Means with K={args.num_modes}...')
    kmeans = KMeans(
        n_clusters=args.num_modes,
        random_state=args.seed,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(flat_trajs)

    # Cluster centers: (K, T*2) -> (K, T, 2)
    centers = kmeans.cluster_centers_.reshape(args.num_modes, T, 2)

    # Print cluster statistics
    labels = kmeans.labels_
    print(f'\nCluster statistics:')
    for k in range(args.num_modes):
        count = (labels == k).sum()
        center_k = centers[k]
        displacement = np.sqrt((center_k ** 2).sum(-1))
        total_disp = displacement[-1]
        mean_speed = total_disp / (T * 0.5)  # assuming 0.5s per step
        print(f'  Mode {k}: {count:5d} trajs ({100*count/M:.1f}%), '
              f'final displacement={total_disp:.2f}m, '
              f'mean speed={mean_speed:.2f}m/s')

    # Save anchors
    np.save(args.output, centers)
    print(f'\nSaved anchors to {args.output}: shape {centers.shape}')

    # Also print anchor summary for verification
    print('\nAnchor final positions (x, y) at last step:')
    for k in range(args.num_modes):
        print(f'  Mode {k}: ({centers[k, -1, 0]:.2f}, {centers[k, -1, 1]:.2f})')


if __name__ == '__main__':
    main()
