"""Wrap a raw OpenCOOD checkpoint for use with CooperativeDetector.

OpenCOOD's `net_epochN.pth` is a bare state_dict with top-level keys like
`pillar_vfe.*`, `backbone.*`, `cls_head.*`. Our wrapper detector holds the
OpenCOOD model under `self.opencood`, so every key needs the `opencood.`
prefix. We also wrap in `{'state_dict': ..., 'meta': ...}` so mmengine's
`--load-from` consumes it directly.

This does NOT do any key remapping or splitting (no backbone/neck split, no
hwl swap, etc.). The OpenCOOD model is imported as-is so it expects the raw
layout.

Usage:
  python tools/wrap_opencood_ckpt.py SRC.pth DST.pth
"""
import argparse
from pathlib import Path

import torch


def wrap(src: Path, dst: Path) -> None:
    obj = torch.load(src, map_location='cpu')
    if isinstance(obj, dict):
        for k in ('model_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict) \
                    and all(isinstance(v, torch.Tensor) for v in obj[k].values()):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError(f'Unexpected checkpoint structure at {src}')

    new_sd = {f'opencood.{k}': v for k, v in obj.items()}

    out = {
        'state_dict': new_sd,
        'meta': {
            'source': str(src),
            'wrapper': 'tools/wrap_opencood_ckpt.py',
        },
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f'Wrote {dst}  ({len(new_sd)} keys, all prefixed with opencood.)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    args = ap.parse_args()
    wrap(args.src, args.dst)


if __name__ == '__main__':
    main()
