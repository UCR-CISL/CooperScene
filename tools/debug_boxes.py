"""One-shot debug: dump predicted vs GT boxes for a single cooperative sample.

Localizes the "3D AP ~ 0 while BEV partially works" gap by printing the actual
box values (x, y, z, l, w, h, yaw) from the mmengine path. Run in-container.

    python tools/debug_boxes.py configs/v2vam/v2vam.py \
        work_dirs/opencood_converted/v2vam.pth \
        --data-root /workspace/data/Cooperscene/release/250928_opv2v \
        --ann-file cooperscene_coop_infos_test_ego1.pkl \
        --num 3
"""
import argparse

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from mmdet3d.registry import MODELS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--ann-file', default='cooperscene_coop_infos_test_ego1.pkl')
    ap.add_argument('--num', type=int, default=2, help='samples to dump')
    return ap.parse_args()


def _np(boxes):
    if boxes is None:
        return np.zeros((0, 7), dtype=np.float32)
    if hasattr(boxes, 'tensor'):
        return boxes.tensor.detach().cpu().numpy()
    if hasattr(boxes, 'cpu'):
        return boxes.cpu().numpy()
    return np.asarray(boxes)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.batch_size = 1

    loader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    it = iter(loader)
    for s in range(args.num):
        batch = next(it)
        with torch.no_grad():
            data = model.data_preprocessor(batch, False)
            # --- fusion sanity: how many CAVs actually go into the model? ---
            inp = data['inputs']
            rl = inp.get('record_len', None)
            coors = inp.get('voxels', {}).get('coors', None)
            cav_counts = None
            if coors is not None:
                cav_counts = [(int(c), int((coors[:, 0] == c).sum()))
                              for c in coors[:, 0].unique().tolist()]
            print(f'  [fusion] record_len={rl.tolist() if hasattr(rl, "tolist") else rl}'
                  f'  per-CAV voxel counts={cav_counts}')
            results = model.predict(inp, data['data_samples'])
        ds = results[0]

        pb = _np(ds.pred_instances_3d.bboxes_3d)
        psc = ds.pred_instances_3d.scores_3d.detach().cpu().numpy()

        # GT may live in gt_instances_3d or eval_ann_info
        gt = None
        if 'gt_instances_3d' in ds and ds.gt_instances_3d is not None \
                and len(getattr(ds.gt_instances_3d, 'bboxes_3d', [])) > 0:
            gt = _np(ds.gt_instances_3d.bboxes_3d)
        elif getattr(ds, 'eval_ann_info', None):
            gt = _np(ds.eval_ann_info.get('gt_bboxes_3d', None))

        print('\n' + '=' * 70)
        print(f'SAMPLE {s}   #pred={len(pb)}  #gt={0 if gt is None else len(gt)}')
        print('  columns = [x, y, z(bottom), l, w, h, yaw]   (mmdet3d LiDAR)')

        if len(pb):
            order = psc.argsort()[::-1][:8]
            print('  -- top pred boxes (by score) --')
            for i in order:
                print('    ', np.round(pb[i], 2), ' score=', round(float(psc[i]), 3))
            print(f'  pred  z(bottom) range: [{pb[:,2].min():.2f}, {pb[:,2].max():.2f}]'
                  f'   h range: [{pb[:,5].min():.2f}, {pb[:,5].max():.2f}]')
        if gt is not None and len(gt):
            print('  -- gt boxes (first 8) --')
            for i in range(min(8, len(gt))):
                print('    ', np.round(gt[i], 2))
            print(f'  gt    z(bottom) range: [{gt[:,2].min():.2f}, {gt[:,2].max():.2f}]'
                  f'   h range: [{gt[:,5].min():.2f}, {gt[:,5].max():.2f}]')

        # nearest-GT pairing in BEV for the top pred, show z gap
        if len(pb) and gt is not None and len(gt):
            top = pb[psc.argsort()[::-1][0]]
            d = np.linalg.norm(gt[:, :2] - top[:2], axis=1)
            j = int(d.argmin())
            print(f'  top-pred vs nearest-GT (bev dist={d[j]:.2f} m):')
            print(f'    pred z_c={top[2]+top[5]/2:.2f} h={top[5]:.2f} | '
                  f'gt z_c={gt[j,2]+gt[j,5]/2:.2f} h={gt[j,5]:.2f} | '
                  f'dz_center={ (top[2]+top[5]/2)-(gt[j,2]+gt[j,5]/2):.2f}')


if __name__ == '__main__':
    main()
