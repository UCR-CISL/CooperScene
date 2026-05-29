"""Anchor-based single-stage detection head (OpenCOOD PointPillar-style).

Input:  BEV feature map x of shape (B, C=in_channels, H, W).
Output: cls map psm (B, A, H, W) + reg map rm (B, 7A, H, W),
        where A = anchor_number (2 rotations on OPV2V).
Loss:   focal (cls) + smooth-L1 with sin-difference on yaw (reg).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes


@MODELS.register_module()
class DetHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 anchor_number=2,
                 anchor_size=[3.9, 1.6, 1.56],
                 anchor_rotations=[0, 90],
                 anchor_z=-1.0,
                 point_cloud_range=[-140.8, -38.4, -3, 140.8, 38.4, 1],
                 voxel_size=[0.4, 0.4, 4],
                 feature_stride=2,
                 pos_threshold=0.6,
                 neg_threshold=0.45,
                 score_threshold=0.20,
                 nms_threshold=0.15,
                 max_num=100,
                 cls_weight=1.0,
                 reg_weight=2.0,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.cls_head = nn.Conv2d(in_channels, anchor_number, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, 7 * anchor_number,
                                  kernel_size=1)

        self.anchor_number = anchor_number
        self.anchor_size = anchor_size
        self.anchor_rotations = [r * np.pi / 180 for r in anchor_rotations]
        self.anchor_z = anchor_z
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_stride = feature_stride
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_num = max_num
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

        self.alpha = 0.25
        self.gamma = 2.0
        self.smooth_l1_beta = 1.0 / 9.0

        self._anchors = None
        self._anchor_cache_key = None

    def _generate_anchors(self, H, W, device):
        cache_key = (H, W, device)
        if self._anchor_cache_key == cache_key and self._anchors is not None:
            return self._anchors

        pcr = self.point_cloud_range
        vw, vh = self.voxel_size[0], self.voxel_size[1]

        # Match OpenCOOD VoxelPostprocessor anchor generation exactly so
        # converted ckpts decode boxes at the same world coordinates.  The
        # offset is the voxel size, independent of feature_stride.
        x = np.linspace(pcr[0] + vw, pcr[3] - vw, W)
        y = np.linspace(pcr[1] + vh, pcr[4] - vh, H)

        xx, yy = np.meshgrid(x, y)
        l, w, h = self.anchor_size

        anchors = []
        for r in self.anchor_rotations:
            anchor = np.stack([
                xx, yy,
                np.full_like(xx, self.anchor_z),
                np.full_like(xx, l),
                np.full_like(xx, w),
                np.full_like(xx, h),
                np.full_like(xx, r)
            ], axis=-1)
            anchors.append(anchor)

        anchors = np.stack(anchors, axis=2)
        anchors = torch.from_numpy(anchors).float().to(device)
        self._anchors = anchors
        self._anchor_cache_key = cache_key
        return anchors

    def _bev_iou(self, boxes_a, boxes_b):
        def get_aabb(boxes):
            x, y = boxes[:, 0], boxes[:, 1]
            l, w = boxes[:, 3], boxes[:, 4]
            r = boxes[:, 6]
            cos_r = torch.abs(torch.cos(r))
            sin_r = torch.abs(torch.sin(r))
            hw = l * sin_r / 2 + w * cos_r / 2
            hl = l * cos_r / 2 + w * sin_r / 2
            return x - hl, y - hw, x + hl, y + hw

        a_x1, a_y1, a_x2, a_y2 = get_aabb(boxes_a)
        b_x1, b_y1, b_x2, b_y2 = get_aabb(boxes_b)

        inter_x1 = torch.max(a_x1.unsqueeze(1), b_x1.unsqueeze(0))
        inter_y1 = torch.max(a_y1.unsqueeze(1), b_y1.unsqueeze(0))
        inter_x2 = torch.min(a_x2.unsqueeze(1), b_x2.unsqueeze(0))
        inter_y2 = torch.min(a_y2.unsqueeze(1), b_y2.unsqueeze(0))

        inter = torch.clamp(inter_x2 - inter_x1, min=0) * \
                torch.clamp(inter_y2 - inter_y1, min=0)

        area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
        area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
        union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter

        return inter / (union + 1e-7)

    def _assign_targets(self, anchors_flat, gt_boxes):
        N = anchors_flat.shape[0]
        cls_targets = torch.zeros(N, device=anchors_flat.device)
        reg_targets = torch.zeros(N, 7, device=anchors_flat.device)
        reg_weights = torch.zeros(N, device=anchors_flat.device)

        if gt_boxes.shape[0] == 0:
            return cls_targets, reg_targets, reg_weights

        iou = self._bev_iou(anchors_flat, gt_boxes)

        best_anchor_per_gt = iou.argmax(dim=0)

        max_iou_per_anchor, best_gt_per_anchor = iou.max(dim=1)

        pos_mask = max_iou_per_anchor > self.pos_threshold
        pos_mask[best_anchor_per_gt] = True

        neg_mask = max_iou_per_anchor < self.neg_threshold

        cls_targets[pos_mask] = 1.0
        reg_weights[pos_mask] = 1.0

        matched_gt = gt_boxes[best_gt_per_anchor[pos_mask]]
        pos_anchors = anchors_flat[pos_mask]
        reg_targets[pos_mask] = self._encode(pos_anchors, matched_gt)

        return cls_targets, reg_targets, reg_weights

    def _encode(self, anchors, gt):
        # anchors / gt stored as (x, y, z, l, w, h, yaw).
        # reg_head channels follow OpenCOOD's `order: hwl` convention so
        # OpenCOOD-trained ckpts (cobevt/v2vam/v2vnet/v2xvit) decode correctly:
        #   ch3 = log(gt_h / anchor_h)
        #   ch4 = log(gt_w / anchor_w)
        #   ch5 = log(gt_l / anchor_l)
        l, w, h = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        diag = torch.sqrt(l ** 2 + w ** 2)

        dx = (gt[:, 0] - anchors[:, 0]) / diag
        dy = (gt[:, 1] - anchors[:, 1]) / diag
        dz = (gt[:, 2] - anchors[:, 2]) / h
        dh = torch.log(gt[:, 5] / h)
        dw = torch.log(gt[:, 4] / w)
        dl = torch.log(gt[:, 3] / l)
        dr = gt[:, 6] - anchors[:, 6]

        return torch.stack([dx, dy, dz, dh, dw, dl, dr], dim=-1)

    def _decode(self, anchors, deltas):
        # Inverse of _encode.  Reg head channels follow OpenCOOD `hwl`:
        #   ch3 -> h-delta, ch4 -> w-delta, ch5 -> l-delta.
        # Output is in mmdet3d's (l, w, h) slot order at [3, 4, 5].
        l, w, h = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        diag = torch.sqrt(l ** 2 + w ** 2)

        x = deltas[:, 0] * diag + anchors[:, 0]
        y = deltas[:, 1] * diag + anchors[:, 1]
        z = deltas[:, 2] * h + anchors[:, 2]
        dh = torch.exp(deltas[:, 3]) * h
        dw = torch.exp(deltas[:, 4]) * w
        dl = torch.exp(deltas[:, 5]) * l
        r = deltas[:, 6] + anchors[:, 6]

        return torch.stack([x, y, z, dl, dw, dh, r], dim=-1)

    def forward(self, x):
        """x: (B, C, H, W) -> psm (B, A, H, W), rm (B, 7A, H, W)."""
        if isinstance(x, (list, tuple)):
            x = x[0]
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        return psm, rm

    def loss(self, x, batch_data_samples, **kwargs):
        psm, rm = self.forward(x)
        B, _, H, W = psm.shape
        device = psm.device

        anchors = self._generate_anchors(H, W, device)
        anchors_flat = anchors.reshape(-1, 7)

        total_cls_loss = psm.new_zeros(())
        total_reg_loss = psm.new_zeros(())

        for b in range(B):
            gt_instances = batch_data_samples[b].gt_instances_3d
            gt_bboxes = gt_instances.bboxes_3d
            if hasattr(gt_bboxes, 'tensor'):
                gt_boxes = gt_bboxes.tensor.to(device)
            else:
                gt_boxes = gt_bboxes.to(device)

            cls_targets, reg_targets, reg_weights = self._assign_targets(
                anchors_flat, gt_boxes)

            pos_count = reg_weights.sum().clamp(min=1.0)

            cls_pred = psm[b].permute(1, 2, 0).reshape(-1)
            cls_loss = self._focal_loss_sum(cls_pred, cls_targets) / pos_count
            total_cls_loss = total_cls_loss + cls_loss

            rm_pred = rm[b].permute(1, 2, 0).reshape(-1, 7)
            rm_pred_sin, reg_targets_sin = self._add_sin_difference(
                rm_pred, reg_targets, dim=6)
            reg_loss = self._smooth_l1(
                rm_pred_sin, reg_targets_sin, reg_weights)
            reg_loss = reg_loss / pos_count
            total_reg_loss = total_reg_loss + reg_loss

        total_cls_loss = total_cls_loss / B * self.cls_weight
        total_reg_loss = total_reg_loss / B * self.reg_weight

        return dict(loss_cls=total_cls_loss, loss_reg=total_reg_loss)

    @staticmethod
    def _add_sin_difference(pred, target, dim=6):
        pred_r = pred[..., dim:dim + 1]
        tgt_r = target[..., dim:dim + 1]
        pred_enc = torch.sin(pred_r) * torch.cos(tgt_r)
        tgt_enc = torch.cos(pred_r) * torch.sin(tgt_r)
        pred_new = torch.cat(
            [pred[..., :dim], pred_enc, pred[..., dim + 1:]], dim=-1)
        tgt_new = torch.cat(
            [target[..., :dim], tgt_enc, target[..., dim + 1:]], dim=-1)
        return pred_new, tgt_new

    def _focal_loss_sum(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1 - pred_sigmoid) + (1 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        return (focal_weight * bce).sum()

    def _smooth_l1(self, pred, target, weights):
        diff = pred - target
        abs_diff = torch.abs(diff)
        smooth = torch.where(
            abs_diff < self.smooth_l1_beta,
            0.5 * diff ** 2 / self.smooth_l1_beta,
            abs_diff - 0.5 * self.smooth_l1_beta)
        return (smooth * weights.unsqueeze(-1)).sum()

    def predict(self, x, batch_data_samples, **kwargs):
        """x: (B, C, H, W) -> list[InstanceData] (len=B) with
        bboxes_3d (LiDAR, 7-dim), scores_3d, labels_3d.
        """
        psm, rm = self.forward(x)
        return self.predict_from_logits(psm, rm, batch_data_samples)

    def predict_from_logits(self, psm, rm, batch_data_samples=None, **kwargs):
        """Same as predict() but skip the internal cls/reg conv layers.

        Used by `OpenCOODCooperativeDetector` which has its OWN cls/reg heads
        inside the imported OpenCOOD model and just needs anchor decoding +
        NMS + InstanceData packing here.
        """
        B, _, H, W = psm.shape
        device = psm.device

        anchors = self._generate_anchors(H, W, device)
        anchors_flat = anchors.reshape(-1, 7)

        results = []
        for b in range(B):
            scores = torch.sigmoid(
                psm[b].permute(1, 2, 0).reshape(-1))
            reg_pred = rm[b].permute(1, 2, 0).reshape(-1, 7)

            mask = scores > self.score_threshold
            scores = scores[mask]
            reg_pred = reg_pred[mask]
            anchors_sel = anchors_flat[mask]

            if scores.shape[0] == 0:
                result = InstanceData()
                result.bboxes_3d = LiDARInstance3DBoxes(
                    torch.zeros(0, 7, device=device),
                    origin=(0.5, 0.5, 0.5))
                result.scores_3d = torch.zeros(0, device=device)
                result.labels_3d = torch.zeros(
                    0, dtype=torch.long, device=device)
                results.append(result)
                continue

            boxes = self._decode(anchors_sel, reg_pred)

            # === bit-match OpenCOOD's VoxelPostprocessor.post_process ===
            # Build (N, 8, 3) corners with OpenCOOD's convention so that
            # `convert_format` (which takes the first 4 corners as the BEV
            # polygon) gets a valid planar quad. Our decoded boxes are in
            # mmdet3d order [x, y, z_center, l, w, h, yaw]; OpenCOOD wants
            # [x, y, z, h, w, l, yaw] for order='hwl'.
            from opencood.utils.box_utils import (
                boxes_to_corners_3d, remove_large_pred_bbx,
                remove_bbx_abnormal_z, get_mask_for_boxes_within_range_torch)
            from opencood.utils.box_utils import nms_rotated as ocd_nms

            boxes_hwl = boxes.clone()
            boxes_hwl[:, 3] = boxes[:, 5]  # h
            boxes_hwl[:, 5] = boxes[:, 3]  # l
            # boxes_hwl[:, 4] == w (unchanged)
            corners = boxes_to_corners_3d(boxes_hwl, order='hwl')

            keep_size = remove_large_pred_bbx(corners)
            keep_z = remove_bbx_abnormal_z(corners)
            keep_pre = torch.logical_and(keep_size, keep_z)
            boxes = boxes[keep_pre]
            scores = scores[keep_pre]
            corners = corners[keep_pre]

            if scores.shape[0] == 0:
                result = InstanceData()
                result.bboxes_3d = LiDARInstance3DBoxes(
                    torch.zeros(0, 7, device=device),
                    origin=(0.5, 0.5, 0.5))
                result.scores_3d = torch.zeros(0, device=device)
                result.labels_3d = torch.zeros(
                    0, dtype=torch.long, device=device)
                results.append(result)
                continue

            # OpenCOOD's nms_rotated takes (N, 8, 3) corners + scores + thr,
            # returns kept indices as a numpy int32 array.
            keep_np = ocd_nms(corners, scores, self.nms_threshold)
            keep = torch.as_tensor(keep_np, dtype=torch.long, device=device)
            keep = keep[:self.max_num]
            boxes = boxes[keep]
            scores = scores[keep]
            corners = corners[keep]

            # Post-NMS range filter against the same GT_RANGE OpenCOOD uses.
            if boxes.shape[0] > 0:
                mask_in_range = get_mask_for_boxes_within_range_torch(corners)
                boxes = boxes[mask_in_range]
                scores = scores[mask_in_range]

            result = InstanceData()
            result.bboxes_3d = LiDARInstance3DBoxes(
                boxes, origin=(0.5, 0.5, 0.5))
            result.scores_3d = scores
            result.labels_3d = torch.zeros(
                len(boxes), dtype=torch.long, device=device)
            results.append(result)

        return results
