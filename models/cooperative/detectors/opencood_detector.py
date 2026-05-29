"""Cooperative detector that delegates the whole encoder/fusion stack to
OpenCOOD's original implementations.

Why this exists
---------------
The plugin-refactor pipeline reimplements PFE / backbone / fusion in
mmdet3d-native form. Subtle numerical differences in voxelization, BatchNorm
cuDNN paths, fusion module structure, etc. accumulate to a 0.05-0.2 mAP gap
when reusing OpenCOOD-trained ckpts. For paper-grade reproduction we instead
**import OpenCOOD's modules directly** through this wrapper.

The wrapper:
  * keeps the existing CooperativeDataset / data preprocessor / metric
    (so we share `tools/test.py`, `tools/train.py` with BEVFusion and other
    mmdet3d-native models);
  * loads raw OpenCOOD `state_dict` (wrap with `tools/wrap_opencood_ckpt.py`);
  * runs OpenCOOD's `PointPillarintermediate<Arch>` forward end-to-end;
  * decodes psm / rm with our DetHead.predict_from_logits.
"""
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.detectors.base import Base3DDetector


_ARCH_TO_MODULE = {
    'v2vam': ('opencood.models.point_pillar_intermediate_V2VAM',
              'PointPillarintermediateV2VAM'),
    'cobevt': ('opencood.models.point_pillar_cobevt', 'PointPillarCoBEVT'),
    'v2vnet': ('opencood.models.point_pillar_v2vnet', 'PointPillarV2VNet'),
    'v2xvit': ('opencood.models.point_pillar_transformer',
               'PointPillarTransformer'),
}


@MODELS.register_module()
class OpenCOODCooperativeDetector(Base3DDetector):
    """Wraps an OpenCOOD intermediate-fusion model end-to-end.

    Args:
        arch: which OpenCOOD model class to instantiate
            ('v2vam', 'cobevt', 'v2vnet', 'v2xvit').
        opencood_args: kwarg dict passed to OpenCOOD model's __init__ as `args`.
            Same structure as the `model.args` block in OpenCOOD's training
            yaml (pillar_vfe / base_bev_backbone / shrink_header /
            compression / fax_fusion / anchor_number / lidar_range / ...).
        bbox_head: built into `self.bbox_head`; we use its
            `predict_from_logits(psm, rm, ...)` for anchor decoding + NMS.
            The bbox_head's own cls/reg conv layers are **not** used.
        max_cav: passed for record_len bookkeeping.
    """

    def __init__(self,
                 arch: str,
                 opencood_args: dict,
                 bbox_head: ConfigType,
                 max_cav: int = 5,
                 anchor_args: Optional[dict] = None,
                 loss_args: Optional[dict] = None,
                 postprocess_args: Optional[dict] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if arch not in _ARCH_TO_MODULE:
            raise ValueError(
                f'arch must be one of {list(_ARCH_TO_MODULE)}, got {arch!r}')
        mod_name, cls_name = _ARCH_TO_MODULE[arch]
        import importlib
        mod = importlib.import_module(mod_name)
        opencood_cls = getattr(mod, cls_name)
        self.arch = arch
        self.max_cav = max_cav
        # the imported OpenCOOD module; ckpts load into self.opencood.*
        self.opencood = opencood_cls(opencood_args)

        # Reuse our DetHead for anchor decoding + InstanceData packing only.
        if 'train_cfg' not in bbox_head and 'test_cfg' not in bbox_head:
            bbox_head = dict(**bbox_head)
            bbox_head['train_cfg'] = train_cfg
            bbox_head['test_cfg'] = test_cfg
        self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Training-time helpers: OpenCOOD VoxelPostprocessor for anchor target
        # assignment + PointPillarLoss. Only constructed when args provided.
        self._post_processor = None
        self._anchor_box_np = None
        self._criterion = None
        if anchor_args is not None and postprocess_args is not None:
            from opencood.data_utils.post_processor.voxel_postprocessor import (
                VoxelPostprocessor)
            params = dict(postprocess_args)
            params.setdefault('core_method', 'VoxelPostprocessor')
            params.setdefault('order', 'hwl')
            params['anchor_args'] = dict(anchor_args)
            self._post_processor = VoxelPostprocessor(params, train=True)
            self._anchor_box_np = self._post_processor.generate_anchor_box()
            self._post_params = params

        if loss_args is not None:
            from opencood.loss.point_pillar_loss import PointPillarLoss
            self._criterion = PointPillarLoss(dict(loss_args))

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def _to_opencood_data_dict(self, batch_inputs_dict: dict) -> dict:
        """Translate mmdet3d-style cooperative inputs to OpenCOOD data_dict.

        Expected `batch_inputs_dict`:
            voxels: dict with 'voxels' (N, max_pts, C),
                    'num_points' (N,), 'coors' (N, 4 = [cav_idx, z, y, x])
            record_len: (B,) int — CAV count per batch sample
        """
        voxels = batch_inputs_dict['voxels']
        record_len = batch_inputs_dict['record_len']
        if not torch.is_tensor(record_len):
            record_len = torch.as_tensor(record_len, dtype=torch.long)

        # Some OpenCOOD models (cobevt / v2x-vit) unconditionally pull
        # `spatial_correction_matrix` from the batch_dict even when they don't
        # use it (it's there for asynchronous time-delay compensation). Pass
        # identity matrices so the KeyError doesn't fire under proj_first.
        device = voxels['voxels'].device
        batch_size = int(record_len.shape[0]) if record_len.ndim == 1 else 1
        spatial_correction_matrix = torch.eye(
            4, device=device, dtype=torch.float32
        ).reshape(1, 1, 4, 4).expand(batch_size, self.max_cav, 4, 4).contiguous()

        # pairwise_t_matrix is also referenced by some fusion modules
        # (v2vnet builds an affine grid from it). For proj_first=True the
        # right value is identity. We clone to guarantee no stride weirdness
        # left over from expand().
        pairwise_t = batch_inputs_dict.get('pairwise_t_matrix', None)
        if pairwise_t is None:
            pairwise_t = torch.zeros(
                batch_size, self.max_cav, self.max_cav, 4, 4,
                device=device, dtype=torch.float32)
            eye = torch.eye(4, device=device, dtype=torch.float32)
            pairwise_t[:] = eye
            pairwise_t = pairwise_t.contiguous()

        # v2x-vit's transformer reads prior_encoding (B, max_cav, 3) =
        # [time_delay, velocity, infra_flag]. Under proj_first + sync inference
        # we pass zeros; if you need real values, build them in the dataset.
        prior_encoding = batch_inputs_dict.get('prior_encoding', None)
        if prior_encoding is None:
            prior_encoding = torch.zeros(
                batch_size, self.max_cav, 3,
                device=device, dtype=torch.float32)

        return {
            'processed_lidar': {
                'voxel_features': voxels['voxels'],
                'voxel_coords': voxels['coors'],
                'voxel_num_points': voxels['num_points'],
            },
            'record_len': record_len,
            'spatial_correction_matrix': spatial_correction_matrix,
            'pairwise_t_matrix': pairwise_t,
            'prior_encoding': prior_encoding,
        }

    def predict(self, batch_inputs_dict: dict,
                batch_data_samples: SampleList, **kwargs) -> SampleList:
        data_dict = self._to_opencood_data_dict(batch_inputs_dict)
        output_dict = self.opencood(data_dict)
        psm = output_dict['psm']  # (B, A, H, W)
        rm = output_dict['rm']    # (B, 7A, H, W)

        results_list = self.bbox_head.predict_from_logits(
            psm, rm, batch_data_samples)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _build_target_dict(self, batch_data_samples: SampleList,
                           device: torch.device) -> dict:
        """Build {targets, pos_equal_one} the way OpenCOOD's PointPillarLoss
        expects, using OpenCOOD's own `VoxelPostprocessor.generate_label`.

        Per-sample GT (from mmdet3d's LiDARInstance3DBoxes, format
        [x, y, z_bottom, l, w, h, yaw]) is converted to OpenCOOD's hwl center
        convention [x, y, z_center, h, w, l, yaw] and padded to `max_num`.
        """
        import numpy as np
        max_num = int(self._post_params.get('max_num', 100))

        pos_list, tgt_list = [], []
        for ds in batch_data_samples:
            if hasattr(ds, 'gt_instances_3d') and \
                    ds.gt_instances_3d is not None and \
                    len(ds.gt_instances_3d.bboxes_3d) > 0:
                gt = ds.gt_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
            else:
                gt = np.zeros((0, 7), dtype=np.float32)

            gt_hwl = gt.copy().astype(np.float64)
            if gt_hwl.shape[0] > 0:
                gt_hwl[:, 2] = gt[:, 2] + gt[:, 5] / 2.0   # z_bottom -> z_center
                gt_hwl[:, 3] = gt[:, 5]                    # h
                gt_hwl[:, 5] = gt[:, 3]                    # l

            padded = np.zeros((max_num, 7), dtype=np.float64)
            mask = np.zeros(max_num, dtype=np.int32)
            n = min(gt_hwl.shape[0], max_num)
            padded[:n] = gt_hwl[:n]
            mask[:n] = 1

            label_dict = self._post_processor.generate_label(
                gt_box_center=padded,
                anchors=self._anchor_box_np,
                mask=mask)
            pos_list.append(label_dict['pos_equal_one'])
            tgt_list.append(label_dict['targets'])

        targets = torch.from_numpy(
            np.stack(tgt_list, axis=0)).float().to(device)
        pos_equal_one = torch.from_numpy(
            np.stack(pos_list, axis=0)).float().to(device)
        return {'targets': targets, 'pos_equal_one': pos_equal_one}

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList, **kwargs) -> dict:
        if self._criterion is None or self._post_processor is None:
            raise RuntimeError(
                'To train OpenCOODCooperativeDetector, pass `anchor_args`, '
                '`postprocess_args` and `loss_args` in the config.')

        data_dict = self._to_opencood_data_dict(batch_inputs_dict)
        output_dict = self.opencood(data_dict)  # {'psm', 'rm'}

        device = output_dict['psm'].device
        target_dict = self._build_target_dict(batch_data_samples, device)

        total_loss = self._criterion(output_dict, target_dict)
        ld = self._criterion.loss_dict
        return {
            'loss': total_loss,
            'conf_loss': ld['conf_loss'].detach(),
            'reg_loss': ld['reg_loss'].detach(),
        }

    def _forward(self, batch_inputs_dict: dict, **kwargs):
        data_dict = self._to_opencood_data_dict(batch_inputs_dict)
        return self.opencood(data_dict)

    def extract_feat(self, batch_inputs_dict: dict, **kwargs):
        raise NotImplementedError(
            'extract_feat not implemented for OpenCOOD wrapper.')
