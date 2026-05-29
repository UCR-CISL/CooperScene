"""Convert OpenCOOD-modified checkpoint to mmdet3d plugin-refactor format.

Key remapping (deterministic, no model dependency):

  pillar_vfe.*                       -> voxel_encoder.*
  backbone.blocks.{i}.{j}.*          -> backbone.blocks.{i}.{j-1}.*   (drop ZeroPad2d at j=0)
  backbone.deblocks.*                -> neck.deblocks.*
  shrink_conv.*                      -> shrink_conv.*                  (unchanged)
  naive_compressor.*                 -> compressor.*                   (cobevt/v2vam/v2xvit only)
  fusion_net.*                       -> fusion_module.*
  cls_head.*                         -> bbox_head.cls_head.*
  reg_head.*                         -> bbox_head.reg_head.*

Output ckpt is wrapped as {'state_dict': ..., 'meta': ...} so mmengine
`load_from=` can consume it directly.

Usage:
  python tools/convert_opencood_ckpt.py SRC.pth DST.pth [--dry-run]
"""
import argparse
import re
from pathlib import Path

import torch


# OpenCOOD block Sequential is [ZeroPad2d, Conv, BN, ReLU, Conv, BN, ReLU, ...]
# mmdet3d SECOND block Sequential is [Conv, BN, ReLU, Conv, BN, ReLU, ...]
# So OpenCOOD inner index j (j>=1) maps to mmdet3d j-1.  j=0 has no params.
_BACKBONE_BLOCK_RE = re.compile(r'^backbone\.blocks\.(\d+)\.(\d+)\.(.+)$')


def remap_key(key: str) -> str | None:
    """Return new key, or None if the key should be dropped (no-param modules)."""
    # 1. pillar_vfe -> voxel_encoder
    if key.startswith('pillar_vfe.'):
        return 'voxel_encoder.' + key[len('pillar_vfe.'):]

    # 2. backbone.deblocks.* -> neck.deblocks.*
    if key.startswith('backbone.deblocks.'):
        return 'neck.deblocks.' + key[len('backbone.deblocks.'):]

    # 3. backbone.blocks.{i}.{j}.{rest} -> backbone.blocks.{i}.{j-1}.{rest}
    m = _BACKBONE_BLOCK_RE.match(key)
    if m:
        i = int(m.group(1))
        j = int(m.group(2))
        rest = m.group(3)
        if j == 0:
            # ZeroPad2d has no params; should not appear in a state_dict at all
            return None
        return f'backbone.blocks.{i}.{j - 1}.{rest}'

    # 4. shrink_conv stays the same
    if key.startswith('shrink_conv.'):
        return key

    # 5. naive_compressor -> compressor
    if key.startswith('naive_compressor.'):
        return 'compressor.' + key[len('naive_compressor.'):]

    # 6. fusion_net -> fusion_module
    if key.startswith('fusion_net.'):
        return 'fusion_module.' + key[len('fusion_net.'):]

    # 7. cls_head / reg_head go under bbox_head
    if key.startswith('cls_head.'):
        return 'bbox_head.cls_head.' + key[len('cls_head.'):]
    if key.startswith('reg_head.'):
        return 'bbox_head.reg_head.' + key[len('reg_head.'):]

    return key  # unrecognized — keep as-is and surface to user


def load_opencood_state_dict(path: Path) -> dict:
    obj = torch.load(path, map_location='cpu')
    # OpenCOOD saves the raw state_dict; some checkpoints may wrap under 'model_dict'.
    if isinstance(obj, dict):
        for wrapper_key in ('model_dict', 'state_dict', 'model'):
            if wrapper_key in obj and isinstance(obj[wrapper_key], dict):
                inner = obj[wrapper_key]
                # Heuristic: looks like a state_dict if values are tensors
                if all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return inner
        # else assume the dict itself is the state_dict
        return obj
    raise ValueError(f'Unexpected checkpoint structure at {path}: {type(obj)}')


def convert(src: Path, dst: Path, dry_run: bool = False) -> None:
    src_sd = load_opencood_state_dict(src)

    new_sd = {}
    unrecognized = []
    dropped = []
    renamed = []  # list of (old, new)

    for k, v in src_sd.items():
        new_k = remap_key(k)
        if new_k is None:
            dropped.append(k)
            continue
        if new_k == k:
            # If we got here and the key is not one of the known passthroughs
            # (shrink_conv.*), flag it
            if not k.startswith('shrink_conv.'):
                unrecognized.append(k)
        else:
            renamed.append((k, new_k))
        new_sd[new_k] = v

    print(f'Source: {src}')
    print(f'  total source keys: {len(src_sd)}')
    print(f'  renamed: {len(renamed)}')
    print(f'  dropped (no-param): {len(dropped)}')
    print(f'  passthrough (shrink_conv.*): '
          f'{sum(1 for k in src_sd if k.startswith("shrink_conv."))}')
    print(f'  unrecognized (kept as-is, please review): {len(unrecognized)}')

    if unrecognized:
        print('\nUNRECOGNIZED keys (passed through unchanged):')
        for k in unrecognized:
            print(f'  {k}')

    # Sample rename audit
    print('\nSample renames:')
    for k_old, k_new in renamed[:8]:
        print(f'  {k_old}\n    -> {k_new}')
    if len(renamed) > 8:
        print(f'  ... ({len(renamed) - 8} more)')

    if dry_run:
        print('\n[dry-run] not writing output.')
        return

    out_obj = {
        'state_dict': new_sd,
        'meta': {
            'converted_from': str(src),
            'converter': 'tools/convert_opencood_ckpt.py',
        },
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, dst)
    print(f'\nWrote: {dst}  ({len(new_sd)} keys)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path, help='Path to OpenCOOD net_epochN.pth')
    ap.add_argument('dst', type=Path, help='Path to write remapped ckpt')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print mapping summary without writing.')
    args = ap.parse_args()
    convert(args.src, args.dst, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
