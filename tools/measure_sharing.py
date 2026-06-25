#!/usr/bin/env python3
"""Measure the ACTUAL per-cooperator sharing size for the sparse-communication
models (ERMVP, CoSDH) by hooking the modules that decide what is transmitted.

Dense-fusion models (v2vnet/v2xvit/v2vam/cobevt) send the full compressed BEV
feature, so their size is computed analytically by tools/calc_sharing.py. ERMVP
and CoSDH instead transmit a sparse subset:
  - ERMVP  : SortSampler keeps top-k tokens (topk_ratio), then clustered
             (cluster_sample_ratio). Deterministic, but measured here too.
  - CoSDH  : Where2comm masks cells by confidence>threshold -> DATA-DEPENDENT
             communication_rate per frame, summed over the multiscale fusions.

Run it like test.py (it reuses the same Runner); it prints the average
per-cooperator MB over the test set plus latency at --throughput Mbps.

    python tools/measure_sharing.py assets/configs/ermvp/ermvp.py assets/configs/ermvp/ermvp.pth \
        --cfg-options test_dataloader.dataset.data_root=/path/to/data
"""
import argparse
import math
from collections import defaultdict

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

BYTES = 4  # fp32


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--throughput', type=float, default=1.6, help='Mbps')
    ap.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    return ap.parse_args()


# id(module) -> [summed_bytes, n_calls]; each module fires once per forward,
# so averaging per module then summing gives total per-cooperator bytes.
ACC = defaultdict(lambda: [0.0, 0])
META = {}


def sampler_hook(cluster_ratio):
    def hook(module, inputs, output):
        src = output[0]                 # (S, B, C): S = top-k tokens per cav
        s_len, _, c = src.shape
        merged = max(math.ceil(s_len * cluster_ratio), 1)
        ACC[('topk', id(module))][0] += s_len * c * BYTES
        ACC[('topk', id(module))][1] += 1
        ACC[('merged', id(module))][0] += merged * c * BYTES
        ACC[('merged', id(module))][1] += 1
        META['model'] = 'ermvp'
    return hook


def where2comm_hook(compression_ratio):
    def hook(module, inputs, output):
        x = inputs[0]                   # (sum_cav, C, H, W); C is DECODED (full)
        c, h, w = x.shape[1], x.shape[-2], x.shape[-1]
        rate = output[1]
        rate = float(rate.float().mean()) if hasattr(rate, 'float') else float(rate)
        cells = rate * h * w           # transmitted cells per cooperator
        # Only C/compression channels actually cross the wire (NaiveCompressor
        # encodes C->C/ratio for transmit, then decodes back to C locally).
        tx_channels = c / compression_ratio if compression_ratio else c
        ACC[('w2c', id(module))][0] += cells * tx_channels * BYTES
        ACC[('w2c', id(module))][1] += 1
        ACC[('rate', id(module))][0] += rate
        ACC[('rate', id(module))][1] += 1
        META['model'] = 'cosdh'
    return hook


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.checkpoint
    cfg.work_dir = './work_dirs/_measure'

    margs = cfg.get('model', {}).get('model_args', {})
    cluster_ratio = margs.get('comm', {}).get('cluster_sample_ratio', 0.2)
    compression = margs.get('compression', 0) or 0   # 0 = no channel compression

    runner = Runner.from_cfg(cfg)

    n = 0
    for m in runner.model.modules():
        cn = type(m).__name__
        if cn == 'SortSampler':
            m.register_forward_hook(sampler_hook(cluster_ratio)); n += 1
        elif cn == 'Where2comm':
            m.register_forward_hook(where2comm_hook(compression)); n += 1
    print(f'[measure] hooked {n} comm module(s); '
          f'cluster_ratio={cluster_ratio} compression={compression}')

    runner.test()

    def avg(kind):
        return sum(s / c for (k, _), (s, c) in ACC.items() if k == kind and c)

    tp = args.throughput
    print('\n' + '=' * 56)
    print(f'Measured per-cooperator sharing  (model={META.get("model","?")})')
    if META.get('model') == 'ermvp':
        for kind in ('topk', 'merged'):
            mb = avg(kind) / 1e6
            lat = mb * 8 / tp * 1000
            print(f'  {kind:7}: {mb:8.4f} MB  ->  {lat:10.0f} ms @ {tp} Mbps')
        print('  (topk = selected tokens; merged = after token clustering)')
    else:
        mb = avg('w2c') / 1e6           # summed over the 3 multiscale fusions
        lat = mb * 8 / tp * 1000
        rate = (sum(s for (k, _), (s, c) in ACC.items() if k == 'rate')
                / max(1, sum(c for (k, _), (s, c) in ACC.items() if k == 'rate')))
        print(f'  avg comm rate (sparsity): {rate:.4f}')
        print(f'  per-coop: {mb:8.4f} MB  ->  {lat:10.0f} ms @ {tp} Mbps')
        print('  (compressed channels C/ratio x spatial rate; fp32)')
    print('=' * 56)


if __name__ == '__main__':
    main()
