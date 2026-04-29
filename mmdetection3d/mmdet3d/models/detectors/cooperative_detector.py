"""Cooperative (multi-agent) 3D detector.

Pipeline per sample: per-CAV voxels -> shared PFE / scatter / backbone / neck
-> optional shrink -> optional compression -> fusion_module (combines CAVs
via record_len + pairwise_t_matrix) -> bbox_head.

Key tensor shapes (intermediate fusion):
- batch_inputs_dict['voxels']: per-CAV voxel dict, batch dim = sum(record_len)
- record_len: (B,) int, number of CAVs per sample
- pairwise_t_matrix: (B, L, L, 4, 4) CAV-to-CAV transforms (L=max_cav)
- backbone output: (sum(N_cav), C, H, W)
- after fusion: (B, C, H, W) ego-view feature map to bbox_head
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .base import Base3DDetector


@MODELS.register_module()
class CooperativeDetector(Base3DDetector):

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 fusion_module: ConfigType,
                 bbox_head: ConfigType,
                 neck: OptConfigType = None,
                 fusion_type: str = 'intermediate',
                 max_cav: int = 5,
                 compression: OptConfigType = None,
                 shrink_header: OptConfigType = None,
                 backbone_fix: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        self.fusion_module = MODELS.build(fusion_module)
        self.fusion_type = fusion_type
        self.max_cav = max_cav

        if compression is not None:
            self.compressor = MODELS.build(compression)
        else:
            self.compressor = None

        if shrink_header is not None:
            if 'type' in shrink_header:
                self.shrink_conv = MODELS.build(shrink_header)
            else:
                self.shrink_conv = nn.Sequential(
                    nn.Conv2d(
                        shrink_header['in_channels'],
                        shrink_header['out_channels'],
                        kernel_size=shrink_header.get('kernel_size', 3),
                        stride=shrink_header.get('stride', 1),
                        padding=shrink_header.get('padding', 1)),
                    nn.BatchNorm2d(shrink_header['out_channels']),
                    nn.ReLU(inplace=True))
        else:
            self.shrink_conv = None

        if 'train_cfg' in bbox_head or 'test_cfg' in bbox_head:
            pass
        else:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone_fix = backbone_fix
        if self.backbone_fix:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        modules = [self.voxel_encoder, self.middle_encoder, self.backbone]
        if self.neck is not None:
            modules.append(self.neck)
        if self.compressor is not None:
            modules.append(self.compressor)
        if self.shrink_conv is not None:
            modules.append(self.shrink_conv)
        modules.append(self.bbox_head)
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_fix:
            self.voxel_encoder.eval()
            self.middle_encoder.eval()
            self.backbone.eval()
            if self.neck is not None:
                self.neck.eval()
            if self.compressor is not None:
                self.compressor.eval()
            if self.shrink_conv is not None:
                self.shrink_conv.eval()
            self.bbox_head.eval()
        return self

    def extract_feat_single(self, voxel_dict: dict) -> Tensor:
        voxel_features = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['num_points'],
            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def extract_feat(self, batch_inputs_dict: dict) -> Tensor:
        voxel_dict = batch_inputs_dict['voxels']
        record_len = batch_inputs_dict.get('record_len', None)
        pairwise_t_matrix = batch_inputs_dict.get(
            'pairwise_t_matrix', None)
        coop_mask = batch_inputs_dict.get('coop_mask', None)

        x = self.extract_feat_single(voxel_dict)
        if isinstance(x, (tuple, list)):
            x = x[-1] if len(x) == 1 else torch.cat(
                [xi for xi in x], dim=1)

        if self.shrink_conv is not None:
            x = self.shrink_conv(x)

        if (self.fusion_type == 'early' or record_len is None or
                (record_len.max().item() <= 1 if torch.is_tensor(record_len)
                 else max(record_len) <= 1)):
            return x

        if self.compressor is not None:
            x = self.compressor(x)

        x = self._apply_fusion(x, record_len, pairwise_t_matrix, coop_mask)

        return x

    def _apply_fusion(self, x, record_len, pairwise_t_matrix, coop_mask):
        from ..cooperative_modules.fuse_modules import (
            SpatialFusion, RandomSumFusion, AttFusion, V2VAttFusion,
            V2VNetFusion, Where2comm, SwapFusionEncoder, V2XTransformer,
            CoAlignFusion)
        from ..cooperative_modules.fuse_modules.fuse_utils import regroup
        from ..cooperative_modules.fuse_modules.coalign_fuse import (
            normalize_pairwise_tfm)

        if isinstance(self.fusion_module,
                      (SpatialFusion, RandomSumFusion, AttFusion,
                       V2VAttFusion)):
            return self.fusion_module(x, record_len)

        if isinstance(self.fusion_module, V2VNetFusion):
            return self.fusion_module(x, record_len, pairwise_t_matrix)

        if isinstance(self.fusion_module, CoAlignFusion):
            _, _, H, W = x.shape
            voxel_size = self.fusion_module.att.sqrt_dim
            normalized = normalize_pairwise_tfm(
                pairwise_t_matrix.clone(), H, W,
                discrete_ratio=0.4, downsample_rate=2)
            return self.fusion_module(x, record_len, normalized)

        if isinstance(self.fusion_module, SwapFusionEncoder):
            from einops import repeat
            regroup_feature, mask = regroup(
                x, record_len, self.max_cav)
            B, L, C, H, W = regroup_feature.shape
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            com_mask = repeat(com_mask,
                              'b h w c l -> b (h new_h) (w new_w) c l',
                              new_h=H, new_w=W)
            return self.fusion_module(regroup_feature, com_mask)

        if isinstance(self.fusion_module, V2XTransformer):
            regroup_feature, mask = regroup(
                x, record_len, self.max_cav)
            B, L, C, H, W = regroup_feature.shape
            regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
            prior_encoding = torch.zeros(
                B, L, H, W, 3, device=x.device, dtype=x.dtype)
            regroup_feature = torch.cat(
                [regroup_feature, prior_encoding], dim=-1)
            if pairwise_t_matrix is not None:
                spatial_correction = pairwise_t_matrix[:, :, 0, :, :]
            else:
                spatial_correction = torch.eye(
                    4, device=x.device).unsqueeze(0).unsqueeze(0).expand(
                    B, L, 4, 4)
            return self.fusion_module(
                regroup_feature, mask, spatial_correction).permute(
                0, 3, 1, 2)

        if isinstance(self.fusion_module, Where2comm):
            psm_single = x.mean(dim=1, keepdim=True)
            return self.fusion_module(
                x, psm_single, record_len, pairwise_t_matrix)[0]

        return self.fusion_module(x, record_len)

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList,
             **kwargs) -> Dict[str, Tensor]:
        x = self.extract_feat(batch_inputs_dict)
        if isinstance(x, Tensor):
            x = [x]
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        x = self.extract_feat(batch_inputs_dict)
        if isinstance(x, Tensor):
            x = [x]
        results_list = self.bbox_head.predict(
            x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[Tensor]]:
        x = self.extract_feat(batch_inputs_dict)
        if isinstance(x, Tensor):
            x = [x]
        results = self.bbox_head.forward(x)
        return results
