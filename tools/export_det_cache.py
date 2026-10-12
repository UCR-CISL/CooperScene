"""Export repo-A (mmdetection3d-opv2v) AssocTrack det caches from the
published CooperScene BEVFusion LIDAR+CAM checkpoints (single AND coop).

Cache format (identical to mmdetection3d-opv2v tools/cache_det_outputs.py):
  {(scenario:str, agent_id:str, ts:int):
      {feat (200,128) f16,   # LAST decoder-layer query features
       pos  (200,2)  f32,    # decoded BEV centers, ego frame (meters)
       score(200,)   f16,    # sigmoid heatmap max over classes (pre-filter)
       boxes(200,7)  f16}}   # [cx, cy, z_bottom, l, w, h, yaw]

Repo A's TransFusionHead stores `self.last_query_feat = query_feat` right
after the last decoder layer ([B, C, P], input of prediction_heads[-1]).
Repo B's head is a sibling implementation without that attribute; the same
tensor is the OUTPUT of `head.decoder[num_decoder_layers-1]` (its forward
transposes back to [B, C, P] before returning), so we grab it with a
forward hook -- byte-identical semantics, no repo-B file modified.

Coop notes:
  * CoopDataset reads OUR motion pkls directly (rows carry `cooperators`
    with agent_id / ego_pose / lidar_points / images).
  * We force agent_setting=None so every motion row yields exactly ONE
    sample with all in-range cooperators (agent_setting='V+2V' would DROP
    rows whose peers are >com_range away -> cache misses; ~25% of test
    rows have a far peer).
  * Infra (agent '0', lidar-only, no camera) is EXCLUDED by default to
    match the Table-4 vehicles-only V/V+V/V+2V protocol (--keep-infra to
    include it as a filler when a peer is out of range).

Usage (GPU, PYTHONPATH=<repo B>):
  python tools/export_det_cache.py \
      --cfg assets/configs/bevfusion/bevfusion_single_lidarcam.py \
      --ckpt assets/configs/bevfusion/bevfusion_single_lidarcam.pth \
      --pkls /data/HangQiu/data/CooperScene/cooperscene_motion_infos_{train,test,val}.pkl \
      --data-root /data/HangQiu/data/CooperScene/ \
      --out <repoA>/work_dirs/assoctrack/det_cache_cs72_lidarcam.pkl
"""
import argparse
import copy
import importlib
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_B)

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import DATASETS, MODELS


def frame_key(info):
    return (str(info.get('scenario')), str(info.get('agent_id')),
            int(round(float(info.get('timestamp')))))


def build_ds(cfg, ann, data_root, is_coop):
    t = copy.deepcopy(cfg.test_dataloader['dataset'])
    t['ann_file'] = ann          # absolute path -> mmengine keeps it as-is
    t['data_root'] = data_root
    t['test_mode'] = True
    if is_coop:
        # One sample per motion-pkl row, all in-range cooperators attached
        # (V+2V normally; degrades gracefully instead of dropping the row).
        t['agent_setting'] = None
    return DATASETS.build(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--pkls', nargs='+', required=True,
                    help='repo-A CooperScene motion info pkls (abs paths)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default='/data/HangQiu/data/CooperScene/')
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke: only N frames per pkl (skips coverage check)')
    ap.add_argument('--keep-infra', action='store_true',
                    help='coop only: allow infra agent "0" as cooperator')
    args = ap.parse_args()

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(args.cfg)
    for m in cfg.get('custom_imports', {}).get('imports', []):
        importlib.import_module(m)

    is_coop = str(cfg.model['type']).startswith('Coop')
    print(f'model={cfg.model["type"]} coop={is_coop}', flush=True)

    if is_coop and not args.keep_infra:
        # Runtime-only patch (this process): vehicles-only cooperators,
        # matching the Table-4 V/V+V/V+2V protocol. No repo file changed.
        from models.cooperative.datasets.coop_dataset import CoopDataset
        orig = CoopDataset._build_valid_coops

        def _vehicles_only(self, info):
            return [c for c in orig(self, info)
                    if str(c.get('agent_id')) != '0']
        CoopDataset._build_valid_coops = _vehicles_only

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model.eval().cuda()
    head = model.bbox_head
    coder = head.bbox_coder
    P = head.num_proposals

    # --- capture the last decoder layer's output query features [B, C, P] ---
    cap = {}
    head.decoder[head.num_decoder_layers - 1].register_forward_hook(
        lambda m, inp, out: cap.__setitem__('q', out))

    cache = {}
    t0 = time.time()
    n_done = 0
    for ann in args.pkls:
        ds = build_ds(cfg, ann, args.data_root, is_coop)
        n = min(len(ds), args.limit) if args.limit else len(ds)
        print(f'[{ann}] samples={len(ds)} running={n}', flush=True)
        for i in range(n):
            info = ds.get_data_info(i)
            key = frame_key(info)
            if i == 0:
                print('  key sample:', key, flush=True)
            if key in cache:
                continue
            batch = pseudo_collate([ds[i]])
            proc = model.data_preprocessor(batch, training=False)
            metas = [s.metainfo for s in proc['data_samples']]
            cap.clear()
            with torch.no_grad():
                feats = model.extract_feat(proc['inputs'], metas)
                preds = head(feats, metas)
            pd = preds[0][0]
            assert 'q' in cap, 'decoder hook did not fire'
            feat = cap['q'][0].permute(1, 0)          # (P, 128)
            assert feat.shape[0] == P, feat.shape

            center = pd['center'][0, :, -P:]
            cx = center[0] * coder.out_size_factor * coder.voxel_size[0] \
                + coder.pc_range[0]
            cy = center[1] * coder.out_size_factor * coder.voxel_size[1] \
                + coder.pc_range[1]
            score = pd['heatmap'][0, :, -P:].sigmoid().max(0).values
            dim = pd['dim'][0, :, -P:].exp()
            height = pd['height'][0, 0, -P:] - dim[2] * 0.5
            rot = torch.atan2(pd['rot'][0, 0, -P:], pd['rot'][0, 1, -P:])
            boxes = torch.stack(
                [cx, cy, height, dim[0], dim[1], dim[2], rot], -1)

            if n_done == 0:
                # one-time equivalence check vs repo-B's own bbox_coder.decode
                ref = coder.decode(
                    pd['heatmap'][..., -P:].sigmoid().clone(),
                    pd['rot'][..., -P:].clone(),
                    pd['dim'][..., -P:].clone(),
                    pd['center'][..., -P:].clone(),
                    pd['height'][..., -P:].clone(),
                    None, filter=False)[0]
                d_box = (ref['bboxes'][:, :7] - boxes).abs().max().item()
                d_sc = (ref['scores'] - score).abs().max().item()
                print(f'  decode check: max|dbox|={d_box:.2e} '
                      f'max|dscore|={d_sc:.2e} feat={tuple(feat.shape)}',
                      flush=True)
                assert d_box < 1e-4 and d_sc < 1e-6

            cache[key] = dict(
                feat=feat.cpu().numpy().astype(np.float16),
                pos=torch.stack([cx, cy], -1).cpu().numpy().astype(np.float32),
                score=score.cpu().numpy().astype(np.float16),
                boxes=boxes.cpu().numpy().astype(np.float16))
            n_done += 1
            if n_done % 500 == 0:
                dt = time.time() - t0
                print(f'  {n_done} frames, {dt/n_done:.2f} s/frame',
                      flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(cache, f)
    dt = time.time() - t0
    print(f'wrote {args.out}: {len(cache)} frames, '
          f'{os.path.getsize(args.out)/2**30:.2f} GB, '
          f'{dt/max(n_done,1):.2f} s/frame avg', flush=True)

    if args.limit:
        print('limit set -> skipping coverage check', flush=True)
        return
    # --- coverage: every motion row must have a cache entry ---
    for p in args.pkls:
        d = pickle.load(open(p, 'rb'))
        dl = d.get('data_list') if isinstance(d, dict) else d
        miss = sum(frame_key(e) not in cache for e in dl)
        tag = 'OK' if miss == 0 else 'MISSING'
        print(f'coverage [{os.path.basename(p)}]: {len(dl)-miss}/{len(dl)} '
              f'{tag}', flush=True)


if __name__ == '__main__':
    main()
