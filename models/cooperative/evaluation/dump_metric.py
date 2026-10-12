"""Dump per-sample predictions, GT and cooperative meta for offline replay.

Used to precompute the demo's "preset model outputs" on the test split: for
every ego sample the detector's boxes, the GT boxes, and which cooperators were
fused with what delay (network='trace' / 'cv2x') are written to one pickle that
the visualizer can replay without a GPU.

Record layout (one dict per data sample, all boxes in the EGO LiDAR frame,
mmdet3d 7-DOF [x, y, z_bottom, l, w, h, yaw]):
  sample_idx, scenario, timestamp, agent_id, ego_pose (6: x y z roll yaw pitch)
  cooperators: [{agent_id, delay_frames}]  (only the ones actually fused)
  transformation_matrix: (max_cav, 4, 4) cooperator -> ego
  pred_bboxes (N, 7) float32, pred_scores (N,), pred_labels (N,)
  gt_bboxes (M, 7) float32, gt_labels (M,)
"""
import os.path as osp
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log

from mmdet3d.registry import METRICS


def _np(x):
    if x is None:
        return None
    if hasattr(x, 'tensor'):
        x = x.tensor
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@METRICS.register_module()
class DumpMetric(BaseMetric):

    def __init__(self,
                 out_file: str,
                 tag: Optional[dict] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        self.default_prefix = 'Dump'
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.out_file = out_file
        self.tag = dict(tag or {})

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for ds in data_samples:
            pred = ds['pred_instances_3d']
            rec = {
                'sample_idx': ds.get('sample_idx'),
                'scenario': ds.get('scenario'),
                'timestamp': ds.get('timestamp'),
                'agent_id': ds.get('agent_id'),
                'ego_pose': ds.get('ego_pose'),
                'cooperators': [
                    {'agent_id': c.get('agent_id'),
                     'delay_frames': int(c.get('delay_frames', 0))}
                    for c in (ds.get('cooperators') or [])],
                # in range but not deliverable within the trace max delay
                'dropped_cooperators': [
                    str(a) for a in (ds.get('dropped_cooperators') or [])],
                'transformation_matrix': _np(ds.get('transformation_matrix')),
                'pred_bboxes': _np(pred['bboxes_3d']).astype(np.float32),
                'pred_scores': _np(pred['scores_3d']).astype(np.float32),
                'pred_labels': _np(pred['labels_3d']),
            }
            gt = ds.get('eval_ann_info') or {}
            if gt and gt.get('gt_bboxes_3d') is not None:
                rec['gt_bboxes'] = _np(gt['gt_bboxes_3d']).astype(np.float32)
                rec['gt_labels'] = _np(gt['gt_labels_3d'])
            elif 'gt_instances_3d' in ds:
                rec['gt_bboxes'] = _np(
                    ds['gt_instances_3d']['bboxes_3d']).astype(np.float32)
                rec['gt_labels'] = _np(ds['gt_instances_3d']['labels_3d'])
            else:
                rec['gt_bboxes'] = np.zeros((0, 7), np.float32)
                rec['gt_labels'] = np.zeros((0,), np.int64)
            self.results.append(rec)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        out_dir = osp.dirname(osp.abspath(self.out_file))
        import os
        os.makedirs(out_dir, exist_ok=True)
        with open(self.out_file, 'wb') as f:
            pickle.dump({'tag': self.tag, 'records': results}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        n_pred = int(sum(len(r['pred_bboxes']) for r in results))
        print_log(f'[DumpMetric] wrote {len(results)} samples / {n_pred} boxes '
                  f'-> {self.out_file} (tag={self.tag})', logger='current')
        return {'num_samples': float(len(results)),
                'num_pred_boxes': float(n_pred)}
