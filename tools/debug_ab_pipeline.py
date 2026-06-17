"""Decisive A/B: feed the SAME (scenario, ts, ego) sample through both the
mmengine CoopDataset pipeline AND the OpenCOOD IntermediateFusionDataset
pipeline, run both through the SAME self.opencood weights, and compare the
model input (voxel_features / record_len) and output (psm / rm).

If psm/rm match  -> input is identical; the gap is in post-process/eval.
If psm/rm differ -> input assembly differs; voxel stats localize it.

Run in-container:
    python tools/debug_ab_pipeline.py configs/v2vam/v2vam.py \
        work_dirs/opencood_converted/v2vam.pth \
        --data-root /workspace/data/Cooperscene/release/250928_opv2v \
        --ann-file cooperscene_coop_infos_test_ego1_fixed.pkl
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
    ap.add_argument('--ann-file', default='cooperscene_coop_infos_test_ego1_fixed.pkl')
    return ap.parse_args()


def vox_stats(name, dd):
    vf = dd['processed_lidar']['voxel_features']
    vc = dd['processed_lidar']['voxel_coords']
    rl = dd['record_len']
    cav = vc[:, 0]
    per = [(int(c), int((cav == c).sum())) for c in cav.unique().tolist()]
    print(f'  [{name}] record_len={rl.tolist()}  total_voxels={vf.shape[0]}'
          f'  per-cav={per}')
    print(f'         voxel_feat mean={vf.float().mean():.4f} std={vf.float().std():.4f}'
          f'  coord ranges z[{int(vc[:,1].min())},{int(vc[:,1].max())}]'
          f' y[{int(vc[:,2].min())},{int(vc[:,2].max())}]'
          f' x[{int(vc[:,3].min())},{int(vc[:,3].max())}]')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # ---- build mmengine model (used for BOTH inputs) ----
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    # ---- mmengine dataset, sample 0 ----
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.batch_size = 1
    mm_loader = Runner.build_dataloader(cfg.test_dataloader)
    mm_ds = mm_loader.dataset
    info0 = mm_ds.get_data_info(0)
    scen = info0.get('scenario'); ts = info0.get('timestamp')
    mm_order = [str(info0.get('agent_id'))] + \
        [str(c.get('agent_id')) for c in info0.get('cooperators', [])]
    print(f'TARGET sample: scenario={scen} ts={ts} (mmengine sample 0)')
    print(f'  mmengine cav order (agent_id): {mm_order}')

    batch = next(iter(mm_loader))
    with torch.no_grad():
        data = model.data_preprocessor(batch, False)
        dd_mm = model._to_opencood_data_dict(data['inputs'])
        out_mm = model.opencood(dd_mm)
    print('\n=== mmengine pipeline ===')
    vox_stats('mm', dd_mm)

    # ---- OpenCOOD dataset, matching (scenario, ts) ----
    from models.cooperative.runner import (_build_hypes,
                                           _make_rotating_ego_dataset_class)
    import os
    hypes = _build_hypes(cfg, for_train=False)
    # test mode reads `validate_dir`; point it at the test split (runner.test
    # does the same swap).
    hypes['validate_dir'] = os.path.join(args.data_root, 'test')
    Rot = _make_rotating_ego_dataset_class()
    ocd_ds = Rot(hypes, visualize=False, train=False, ego_candidates=['1'])

    # map (scenario folder name, ts) -> base idx
    # scenarios are folders sorted; scenario_database keyed 0..N in that order.
    import os
    root = hypes['validate_dir']
    scen_names = sorted([x for x in os.listdir(root)
                         if os.path.isdir(os.path.join(root, x))])
    si = scen_names.index(str(scen))
    base_before = ocd_ds.len_record[si - 1] if si > 0 else 0
    # timestamps for that scenario (first cav in db)
    first_cav = next(iter(ocd_ds.scenario_database[si]))
    tss = sorted([k for k in ocd_ds.scenario_database[si][first_cav]
                  if k != 'ego'])
    # ts stored as string like '487850'; pkl ts is float
    ts_str = str(int(ts))
    k = tss.index(ts_str)
    base_idx = base_before + k
    print(f'\nOpenCOOD base_idx={base_idx} (scenario {scen} pos {k})')
    ocd_ds._mark_ego(base_idx, '1')
    base_dd = ocd_ds.retrieve_base_data(base_idx)
    print(f'  OpenCOOD cav order (cav_id): {list(base_dd.keys())}  '
          f'ego flags={[base_dd[c]["ego"] for c in base_dd]}')

    batch_ocd = ocd_ds.collate_batch_test([ocd_ds[base_idx]])
    ego = batch_ocd['ego']

    def _cuda(x):
        return x.cuda() if torch.is_tensor(x) else x
    pl = ego['processed_lidar']
    dd_ocd = {
        'processed_lidar': {
            'voxel_features': _cuda(pl['voxel_features']),
            'voxel_coords': _cuda(pl['voxel_coords']),
            'voxel_num_points': _cuda(pl['voxel_num_points']),
        },
        'record_len': _cuda(ego['record_len']),
        'spatial_correction_matrix': _cuda(ego.get('spatial_correction_matrix')),
        'pairwise_t_matrix': _cuda(ego.get('pairwise_t_matrix')),
        'prior_encoding': _cuda(ego.get('prior_encoding')),
    }
    with torch.no_grad():
        out_ocd = model.opencood(dd_ocd)
    print('\n=== OpenCOOD pipeline ===')
    vox_stats('ocd', dd_ocd)

    # ---- compare psm / rm ----
    print('\n=== model output (psm / rm) ===')
    psm_mm, psm_o = out_mm['psm'], out_ocd['psm']
    rm_mm, rm_o = out_mm['rm'], out_ocd['rm']
    print(f'  psm shape mm={tuple(psm_mm.shape)} ocd={tuple(psm_o.shape)}')
    if psm_mm.shape == psm_o.shape:
        print(f'  psm max diff={ (psm_mm-psm_o).abs().max():.4e}'
              f'  | mm sig>0.2: {(psm_mm.sigmoid()>0.2).sum().item()}'
              f'  ocd sig>0.2: {(psm_o.sigmoid()>0.2).sum().item()}')
        print(f'  rm  max diff={ (rm_mm-rm_o).abs().max():.4e}')
    print(f'  psm.sigmoid max: mm={psm_mm.sigmoid().max():.3f} ocd={psm_o.sigmoid().max():.3f}')

    # ---- compare FINAL predictions: DetHead vs OpenCOOD post_process ----
    # feed the SAME psm/rm (mmengine's) to both post-processors.
    print('\n=== post-process A/B (same psm/rm = mmengine out) ===')
    with torch.no_grad():
        res = model.bbox_head.predict_from_logits(
            psm_mm, rm_mm, data['data_samples'])
    pb = res[0].bboxes_3d.tensor.cpu().numpy()
    ps = res[0].scores_3d.cpu().numpy()
    print(f'  DetHead: #pred={len(pb)}  scores[min/max]='
          f'{(ps.min() if len(ps) else 0):.3f}/{(ps.max() if len(ps) else 0):.3f}')

    from opencood.data_utils.post_processor.voxel_postprocessor import (
        VoxelPostprocessor)
    pp_params = dict(
        core_method='VoxelPostprocessor', order='hwl',
        max_num=int(cfg.opencood_postprocess_args['max_num']),
        nms_thresh=float(cfg.opencood_postprocess_args['nms_thresh']),
        anchor_args=dict(cfg.opencood_anchor_args),
        target_args=dict(cfg.opencood_postprocess_args['target_args']),
    )
    pp = VoxelPostprocessor(pp_params, train=False)
    anchor = torch.from_numpy(pp.generate_anchor_box()).float().cuda()
    dd = {'ego': {'anchor_box': anchor,
                  'transformation_matrix': torch.eye(4).cuda()}}
    od = {'ego': {'psm': psm_mm, 'rm': rm_mm}}
    with torch.no_grad():
        pred_box, pred_score = pp.post_process(dd, od)
    nb = 0 if pred_box is None else len(pred_box)
    if nb:
        smin = float(pred_score.min()); smax = float(pred_score.max())
    else:
        smin = smax = 0.0
    print(f'  OpenCOOD post_process(psm_mm): #pred={nb}  scores[min/max]={smin:.3f}/{smax:.3f}')

    # real OpenCOOD prediction: post_process on OpenCOOD's own psm_o
    od2 = {'ego': {'psm': psm_o, 'rm': rm_o}}
    with torch.no_grad():
        pb2, ps2 = pp.post_process(dd, od2)
    nb2 = 0 if pb2 is None else len(pb2)
    s2 = (float(ps2.min()), float(ps2.max())) if nb2 else (0.0, 0.0)
    print(f'  [REAL] mmengine pred (DetHead/psm_mm): #pred={len(pb)}'
          f'  scores={ps.min() if len(ps) else 0:.3f}/{ps.max() if len(ps) else 0:.3f}')
    print(f'  [REAL] OpenCOOD pred (post/psm_o):     #pred={nb2}'
          f'  scores={s2[0]:.3f}/{s2[1]:.3f}')


if __name__ == '__main__':
    main()
