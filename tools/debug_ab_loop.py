"""Loop A/B over N samples: for each mmengine sample, find the matching
OpenCOOD sample (same scenario, ts, ego=1), run both through the SAME
self.opencood, and compare #predictions + record_len + voxel count.

Localizes whether the residual bs=1 gap is per-sample (which samples diverge)
or aggregate.

    python tools/debug_ab_loop.py configs/v2vam/v2vam.py \
        work_dirs/opencood_converted/v2vam.pth \
        --data-root /workspace/data/Cooperscene/release/250928_opv2v \
        --ann-file cooperscene_coop_infos_test_ego1_fixed.pkl --num 30
"""
import argparse
import os
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
    ap.add_argument('--num', type=int, default=30)
    return ap.parse_args()


def npred(model, dd):
    out = model.opencood(dd)
    # decode via DetHead, count predictions
    res = model.bbox_head.predict_from_logits(out['psm'], out['rm'], None)
    return len(res[0].bboxes_3d.tensor), out['psm']


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.batch_size = 1
    mm_loader = Runner.build_dataloader(cfg.test_dataloader)
    mm_ds = mm_loader.dataset

    from models.cooperative.runner import (_build_hypes,
                                           _make_rotating_ego_dataset_class)
    hypes = _build_hypes(cfg, for_train=False)
    hypes['validate_dir'] = os.path.join(args.data_root, 'test')
    ocd_ds = _make_rotating_ego_dataset_class()(
        hypes, visualize=False, train=False, ego_candidates=['1'])
    scen_names = sorted([x for x in os.listdir(hypes['validate_dir'])
                         if os.path.isdir(os.path.join(hypes['validate_dir'], x))])

    def ocd_dd_for(scen, ts):
        si = scen_names.index(str(scen))
        base_before = ocd_ds.len_record[si - 1] if si > 0 else 0
        first_cav = next(iter(ocd_ds.scenario_database[si]))
        tss = sorted([k for k in ocd_ds.scenario_database[si][first_cav]
                      if k != 'ego'])
        bidx = base_before + tss.index(str(int(ts)))
        ego = ocd_ds.collate_batch_test([ocd_ds[bidx]])['ego']
        pl = ego['processed_lidar']
        c = lambda x: x.cuda() if torch.is_tensor(x) else x
        return {'processed_lidar': {k: c(pl[k]) for k in pl},
                'record_len': c(ego['record_len']),
                'spatial_correction_matrix': c(ego.get('spatial_correction_matrix')),
                'pairwise_t_matrix': c(ego.get('pairwise_t_matrix')),
                'prior_encoding': c(ego.get('prior_encoding'))}

    print(f'{"idx":>3} {"scen":>5} {"ts":>9} {"rl_mm":>5} {"rl_o":>5} '
          f'{"np_mm":>5} {"np_o":>5} {"psmΔ":>7}')
    n_mismatch = 0
    for i in range(args.num):
        info = mm_ds.get_data_info(i)
        scen, ts = info.get('scenario'), info.get('timestamp')
        batch = next(iter(torch.utils.data.DataLoader(
            mm_ds, batch_size=1, num_workers=0,
            collate_fn=mm_loader.collate_fn,
            sampler=[i])))
        with torch.no_grad():
            data = model.data_preprocessor(batch, False)
            dd_mm = model._to_opencood_data_dict(data['inputs'])
            np_mm, psm_mm = npred(model, dd_mm)
            rl_mm = int(dd_mm['record_len'].sum())
            try:
                dd_o = ocd_dd_for(scen, ts)
                np_o, psm_o = npred(model, dd_o)
                rl_o = int(dd_o['record_len'].sum())
                pd = float((psm_mm - psm_o).abs().max()) if psm_mm.shape == psm_o.shape else -1
            except Exception as e:
                np_o, rl_o, pd = -1, -1, -1
        flag = '' if (np_mm == np_o and rl_mm == rl_o) else '  <<DIFF'
        if flag:
            n_mismatch += 1
        print(f'{i:>3} {str(scen):>5} {str(int(ts)):>9} {rl_mm:>5} {rl_o:>5} '
              f'{np_mm:>5} {np_o:>5} {pd:>7.3f}{flag}')
    print(f'\nmismatched samples: {n_mismatch}/{args.num}')


if __name__ == '__main__':
    main()
