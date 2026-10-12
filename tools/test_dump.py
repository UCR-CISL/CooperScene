"""Evaluate a cooperative model AND dump its per-sample outputs for replay.

Same CLI as tools/test.py plus --out. Keeps the config's EvalMetric (so the
mAP numbers are still logged) and appends a DumpMetric that writes one pickle
with predictions, GT and the fused-cooperator meta (agent ids, delays) per
ego sample. Intended to precompute the demo's preset outputs on the test split:

  python tools/test_dump.py assets/configs/cosdh/cosdh.py assets/configs/cosdh/cosdh.pth \
      --out work_dirs/dumps/cosdh__V-2V-I__trace30.pkl \
      --cfg-options test_dataloader.dataset.agent_setting=V+2V+I \
                    test_dataloader.dataset.network=trace \
                    test_dataloader.dataset.share_size_mb=0.0595 \
                    test_dataloader.dataset.trace_max_delay=30
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    p = argparse.ArgumentParser(description='Test a model and dump outputs')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--out', required=True, help='output pickle path')
    p.add_argument('--work-dir')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                   default='none')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = p.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    ds = cfg.test_dataloader.dataset
    tag = dict(
        config=osp.basename(args.config),
        checkpoint=osp.basename(args.checkpoint),
        agent_setting=ds.get('agent_setting'),
        network=ds.get('network', 'unlimited'),
        share_size_mb=ds.get('share_size_mb'),
        cv2x_throughput=ds.get('cv2x_throughput'),
        trace_max_delay=ds.get('trace_max_delay'),
        trace_drop=ds.get('trace_drop'),
        ann_file=ds.get('ann_file'),
    )
    evaluators = cfg.test_evaluator
    if isinstance(evaluators, dict):
        evaluators = [evaluators]
    evaluators = list(evaluators) + [
        dict(type='DumpMetric', out_file=args.out, tag=tag)]
    cfg.test_evaluator = evaluators

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
