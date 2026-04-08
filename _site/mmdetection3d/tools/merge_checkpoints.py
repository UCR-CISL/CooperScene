"""Merge two BEVFusion checkpoints for cooperative LiDAR+Camera model.

Merges weights from:
  1. Single-agent lidar+cam checkpoint (camera branch, pts backbone/neck)
  2. Cooperative lidar checkpoint (SwapFusion, compress/expand, bbox_head)

Usage:
    python tools/merge_checkpoints.py \
        --lidarcam path/to/lidarcam_ckpt.pth \
        --coop-lidar path/to/coop_lidar_ckpt.pth \
        --out path/to/merged_ckpt.pth

Module source mapping:
    img_backbone.*, img_neck.*, view_transform.*, fusion_layer.*  <- lidarcam
    pts_voxel_*, pts_middle_encoder.*                              <- either
    pts_backbone.*, pts_neck.*                                     <- lidarcam
    coop_fusion.*, compress.*, expand.*                            <- coop-lidar
    bbox_head.*                                                    <- coop-lidar
"""

import argparse

import torch


def merge_checkpoints(lidarcam_path, coop_lidar_path, out_path):
    """Merge two checkpoint state_dicts by prefix.

    Args:
        lidarcam_path: Path to single-agent lidar+cam checkpoint.
        coop_lidar_path: Path to cooperative lidar checkpoint.
        out_path: Output path for merged checkpoint.
    """
    print(f'Loading lidar+cam checkpoint: {lidarcam_path}')
    lidarcam_ckpt = torch.load(lidarcam_path, map_location='cpu')
    lidarcam_sd = lidarcam_ckpt.get('state_dict', lidarcam_ckpt)

    print(f'Loading coop lidar checkpoint: {coop_lidar_path}')
    coop_ckpt = torch.load(coop_lidar_path, map_location='cpu')
    coop_sd = coop_ckpt.get('state_dict', coop_ckpt)

    # Define which checkpoint provides each module
    # Keys from lidarcam checkpoint:
    lidarcam_prefixes = [
        'img_backbone.',
        'img_neck.',
        'view_transform.',
        'fusion_layer.',
        'pts_backbone.',
        'pts_neck.',
    ]
    # Keys from either (prefer lidarcam since it has fused-feature training)
    either_prefixes = [
        'pts_voxel_layer.',
        'pts_voxel_encoder.',
        'pts_middle_encoder.',
    ]
    # Keys from coop checkpoint:
    coop_prefixes = [
        'coop_fusion.',
        'compress.',
        'expand.',
        'bbox_head.',
    ]
    # Skip (no learned parameters):
    skip_prefixes = [
        'sttf.',
    ]

    merged_sd = {}
    used_from_lidarcam = 0
    used_from_coop = 0
    skipped = 0
    missing = 0

    # Collect all expected keys from both checkpoints
    all_keys = set()
    all_keys.update(lidarcam_sd.keys())
    all_keys.update(coop_sd.keys())

    for key in sorted(all_keys):
        # Check which source this key belongs to
        source = None

        for prefix in lidarcam_prefixes:
            if key.startswith(prefix):
                source = 'lidarcam'
                break

        if source is None:
            for prefix in either_prefixes:
                if key.startswith(prefix):
                    source = 'either'
                    break

        if source is None:
            for prefix in coop_prefixes:
                if key.startswith(prefix):
                    source = 'coop'
                    break

        if source is None:
            for prefix in skip_prefixes:
                if key.startswith(prefix):
                    source = 'skip'
                    break

        if source == 'lidarcam':
            if key in lidarcam_sd:
                merged_sd[key] = lidarcam_sd[key]
                used_from_lidarcam += 1
            else:
                print(f'  WARNING: {key} not found in lidarcam checkpoint')
                missing += 1
        elif source == 'either':
            if key in lidarcam_sd:
                merged_sd[key] = lidarcam_sd[key]
                used_from_lidarcam += 1
            elif key in coop_sd:
                merged_sd[key] = coop_sd[key]
                used_from_coop += 1
            else:
                print(f'  WARNING: {key} not found in either checkpoint')
                missing += 1
        elif source == 'coop':
            if key in coop_sd:
                merged_sd[key] = coop_sd[key]
                used_from_coop += 1
            else:
                print(f'  WARNING: {key} not found in coop checkpoint')
                missing += 1
        elif source == 'skip':
            skipped += 1
        else:
            # Unknown prefix - try to include from whichever has it
            if key in coop_sd:
                merged_sd[key] = coop_sd[key]
                used_from_coop += 1
                print(f'  INFO: {key} (unknown prefix) taken from coop')
            elif key in lidarcam_sd:
                merged_sd[key] = lidarcam_sd[key]
                used_from_lidarcam += 1
                print(f'  INFO: {key} (unknown prefix) taken from lidarcam')

    print(f'\nMerge summary:')
    print(f'  From lidarcam: {used_from_lidarcam} keys')
    print(f'  From coop:     {used_from_coop} keys')
    print(f'  Skipped:       {skipped} keys')
    print(f'  Missing:       {missing} keys')
    print(f'  Total merged:  {len(merged_sd)} keys')

    # Save as a proper mmengine checkpoint
    merged_ckpt = {
        'state_dict': merged_sd,
        'meta': {
            'merged_from': {
                'lidarcam': lidarcam_path,
                'coop_lidar': coop_lidar_path,
            }
        }
    }
    torch.save(merged_ckpt, out_path)
    print(f'\nSaved merged checkpoint to: {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge BEVFusion checkpoints for cooperative LiDAR+Cam')
    parser.add_argument(
        '--lidarcam',
        type=str,
        required=True,
        help='Path to single-agent lidar+cam checkpoint')
    parser.add_argument(
        '--coop-lidar',
        type=str,
        required=True,
        help='Path to cooperative lidar checkpoint')
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output path for merged checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merge_checkpoints(args.lidarcam, args.coop_lidar, args.out)
