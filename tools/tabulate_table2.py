#!/usr/bin/env python3
"""Tabulate tools/run_table2.sh outputs: BEV AP@0.3/0.5/0.7 per model x agent
setting x network (unlimited / cv2x / trace<maxdelay>), plus the trace link
statistics logged by CoopDataset.

    python tools/tabulate_table2.py                      # work_dirs/table2
    python tools/tabulate_table2.py work_dirs/table2 work_dirs/table2_papershare
    python tools/tabulate_table2.py --metric 3D          # AP_3D instead of AP_BEV
    python tools/tabulate_table2.py --md                 # markdown table
"""
import argparse
import glob
import json
import os
import os.path as osp
import re

MODELS = ['v2vnet', 'v2xvit', 'v2vam', 'cobevt', 'ermvp', 'cosdh']
SETTINGS = ['V-I', 'V-V', 'V-V-I', 'V-2V', 'V-2V-I']


def latest_metrics(cell_dir):
    js = sorted(glob.glob(osp.join(cell_dir, '*', '*.json')))
    if not js:
        return None
    with open(js[-1]) as f:
        return json.load(f)


def link_stats(cell_dir):
    """Last '[CoopDataset trace] ...' line of the cell's mmengine log."""
    logs = sorted(glob.glob(osp.join(cell_dir, '*', '*.log')))
    if not logs:
        return ''
    stat = ''
    with open(logs[-1], errors='ignore') as f:
        for line in f:
            if '[CoopDataset trace]' in line:
                stat = line.split('[CoopDataset trace]', 1)[1].strip()
    m = re.search(r'delivered \d+ \(([\d.]+)%\).*?dropped \d+ \(([\d.]+)%\)', stat)
    return f'deliv {m.group(1)}% drop {m.group(2)}%' if m else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='*', default=['work_dirs/table2'])
    ap.add_argument('--metric', choices=['BEV', '3D'], default='BEV')
    ap.add_argument('--md', action='store_true')
    args = ap.parse_args()

    cells = {}
    for root in args.roots:
        for d in sorted(glob.glob(osp.join(root, '*__*__*'))):
            if not osp.isdir(d):
                continue
            model, setting, net = osp.basename(d).split('__')
            m = latest_metrics(d)
            cells[(root, model, setting, net)] = (m, d)
    nets = sorted({k[3] for k in cells},
                  key=lambda n: (['unlimited', 'cv2x'].index(n) if n in ('unlimited', 'cv2x') else 2, n))
    key = f'Eval metric/AP_{args.metric}@'

    def fmt(m):
        if m is None:
            return 'running'
        return '/'.join(f"{m[key + t]:.2f}" for t in ('0.3', '0.5', '0.7'))

    for root in args.roots:
        print(f'\n## {root}  (AP_{args.metric} @0.3/0.5/0.7)')
        hdr = ['model', 'setting'] + nets + ['trace links']
        if args.md:
            print('| ' + ' | '.join(hdr) + ' |')
            print('|' + '---|' * len(hdr))
        else:
            print(''.join(f'{h:>18s}' for h in hdr))
        for model in MODELS:
            for s in SETTINGS:
                row = [model, s]
                stats = []
                any_cell = False
                for n in nets:
                    c = cells.get((root, model, s, n))
                    if c is None:
                        row.append('-')
                        continue
                    any_cell = True
                    row.append(fmt(c[0]))
                    if n.startswith('trace'):
                        ls = link_stats(c[1])
                        if ls:
                            stats.append(f'{n}: {ls}')
                if not any_cell:
                    continue
                row.append('; '.join(stats))
                if args.md:
                    print('| ' + ' | '.join(row) + ' |')
                else:
                    print(''.join(f'{v:>18s}' for v in row[:-1]) + '   ' + row[-1])


if __name__ == '__main__':
    main()
