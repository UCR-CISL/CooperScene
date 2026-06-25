#!/usr/bin/env python3
"""Compute per-cooperator / per-setting sharing size and C-V2X latency.

The transmitted tensor of each intermediate-fusion model is the compressed BEV
feature: NaiveCompressor(256, compression) emits ``256 // compression`` channels
(or the full 256 when compression == 0) at the shrink-output resolution
``(W/stride, H/stride)``, stored as fp32 (4 bytes). Hence

    per_coop_MB = (256//compression or 256) * (W/stride) * (H/stride) * 4 / 1e6
    sharing(setting) = per_coop_MB * num_cooperators(setting)
    latency_ms       = sharing_MB * 8 / throughput_Mbps * 1000

Params below are read from assets/configs/<model>/<model>.py
(point_cloud_range, voxel_size, feature_stride, compression).

Usage:
    python tools/calc_sharing.py                      # full table + Table-2 check
    python tools/calc_sharing.py --model v2vam --percoop   # just the MB number
"""
import argparse

#         xmin   xmax   ymin   ymax  vx   vy  stride comp
MODELS = {
    'v2vnet': (-140.8, 140.8, -40.0, 40.0, 0.4, 0.4, 4, 0),
    'v2xvit': (-140.8, 140.8, -38.4, 38.4, 0.4, 0.4, 4, 32),
    'v2vam':  (-140.8, 140.8, -40.0, 40.0, 0.4, 0.4, 4, 32),
    'cobevt': (-140.8, 140.8, -38.4, 38.4, 0.4, 0.4, 2, 64),
    'cosdh':  (-140.8, 140.8, -38.4, 38.4, 0.4, 0.4, 2, 16),
    'ermvp':  (-140.8, 140.8, -38.4, 38.4, 0.4, 0.4, 4, 0),
}
# agent setting -> number of transmitting cooperators
NCOOP = {'V+I': 1, 'V+V': 1, 'V+V+I': 2, 'V+2V': 2, 'V+2V+I': 3}
# published per-cooperator sharing (MB) from paper Table 2, for cross-check
PAPER_PERCOOP = {'v2vnet': 10.0, 'v2xvit': 0.338, 'v2vam': 0.423, 'cobevt': 0.500}


def per_coop_mb(model):
    xmn, xmx, ymn, ymx, vx, vy, st, comp = MODELS[model]
    w = round((xmx - xmn) / vx) // st
    h = round((ymx - ymn) / vy) // st
    ch = 256 // comp if comp > 0 else 256
    return w * h * ch * 4 / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS))
    ap.add_argument('--setting', choices=list(NCOOP))
    ap.add_argument('--throughput', type=float, default=1.6, help='Mbps')
    ap.add_argument('--percoop', action='store_true',
                    help='print only the per-cooperator MB for --model')
    args = ap.parse_args()

    if args.percoop and args.model:
        print(f'{per_coop_mb(args.model):.3f}')
        return

    tp = args.throughput
    print(f'Per-cooperator sharing (MB) vs paper Table 2:')
    print(f"  {'model':8}{'calc':>8}{'paper':>8}{'ratio':>8}")
    for m in MODELS:
        c = per_coop_mb(m)
        p = PAPER_PERCOOP.get(m)
        r = f'{c/p:.2f}' if p else '-'
        ps = f'{p:.3f}' if p else '-'
        print(f"  {m:8}{c:>8.3f}{ps:>8}{r:>8}")

    print(f'\nSharing (MB) / latency (ms) @ {tp} Mbps:')
    hdr = ''.join(f'{s:>18}' for s in NCOOP)
    print(f"  {'model':8}{hdr}")
    for m in MODELS:
        pc = per_coop_mb(m)
        cells = ''
        for s, n in NCOOP.items():
            share = pc * n
            lat = share * 8 / tp * 1000
            cells += f'{share:8.3f}/{lat:8.0f}'
        print(f"  {m:8}{cells}")


if __name__ == '__main__':
    main()
