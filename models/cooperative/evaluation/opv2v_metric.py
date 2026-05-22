# Copyright (c) OpenMMLab. All rights reserved.
"""OPV2V Evaluation Metric.

Computes both 2D BEV AP and 3D AP for OPV2V dataset.
- 2D BEV AP: Matches original OpenCOOD evaluation (only XY plane IoU)
- 3D AP: Considers full 3D IoU (BEV IoU * height overlap)

Both use VOC 2010 AP calculation method.

Optimized with CUDA-accelerated IoU computation.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes


def compute_bev_iou_cuda(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Compute BEV IoU using CUDA-accelerated rotated box IoU.

    Args:
        pred_boxes: (N, 7) tensor [x, y, z, dx, dy, dz, yaw]
        gt_boxes: (M, 7) tensor [x, y, z, dx, dy, dz, yaw]

    Returns:
        (N, M) IoU matrix
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]),
                          device=pred_boxes.device)

    # Convert to BEV format for rotated IoU: [x, y, dx, dy, yaw]
    pred_bev = torch.cat([
        pred_boxes[:, 0:2],   # x, y
        pred_boxes[:, 3:5],   # dx, dy
        pred_boxes[:, 6:7],   # yaw
    ], dim=1)  # (N, 5)

    gt_bev = torch.cat([
        gt_boxes[:, 0:2],
        gt_boxes[:, 3:5],
        gt_boxes[:, 6:7],
    ], dim=1)  # (M, 5)

    # Use mmcv's box_iou_rotated (CUDA accelerated)
    try:
        from mmcv.ops import box_iou_rotated
        iou_matrix = box_iou_rotated(pred_bev, gt_bev, aligned=False)
    except ImportError:
        # Fallback to mmdet3d's implementation
        pred_boxes_3d = LiDARInstance3DBoxes(pred_boxes)
        gt_boxes_3d = LiDARInstance3DBoxes(gt_boxes)
        iou_matrix = pred_boxes_3d.overlaps(pred_boxes_3d, gt_boxes_3d, mode='iou')

    return iou_matrix


def compute_3d_iou_cuda(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                        bev_iou_matrix: torch.Tensor) -> torch.Tensor:
    """Compute 3D IoU using BEV IoU and height overlap (vectorized).

    Args:
        pred_boxes: (N, 7) tensor [x, y, z, dx, dy, dz, yaw]
        gt_boxes: (M, 7) tensor
        bev_iou_matrix: (N, M) BEV IoU matrix

    Returns:
        (N, M) 3D IoU matrix
    """
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), device=pred_boxes.device)

    # Compute BEV areas
    pred_areas = pred_boxes[:, 3] * pred_boxes[:, 4]  # (N,)
    gt_areas = gt_boxes[:, 3] * gt_boxes[:, 4]  # (M,)

    # Compute intersection areas from BEV IoU
    # IoU = inter / (area1 + area2 - inter)
    # inter = IoU * (area1 + area2) / (1 + IoU)
    sum_areas = pred_areas[:, None] + gt_areas[None, :]  # (N, M)
    inter_areas = bev_iou_matrix * sum_areas / (1 + bev_iou_matrix + 1e-10)

    # Z ranges (center +/- half height)
    pred_z_min = pred_boxes[:, 2] - pred_boxes[:, 5] / 2  # (N,)
    pred_z_max = pred_boxes[:, 2] + pred_boxes[:, 5] / 2
    gt_z_min = gt_boxes[:, 2] - gt_boxes[:, 5] / 2  # (M,)
    gt_z_max = gt_boxes[:, 2] + gt_boxes[:, 5] / 2

    # Z overlap
    z_overlap = torch.clamp(
        torch.min(pred_z_max[:, None], gt_z_max[None, :]) -
        torch.max(pred_z_min[:, None], gt_z_min[None, :]),
        min=0)  # (N, M)

    # 3D volumes
    pred_volumes = pred_areas * pred_boxes[:, 5]  # (N,)
    gt_volumes = gt_areas * gt_boxes[:, 5]  # (M,)

    # 3D intersection and union
    inter_volumes = inter_areas * z_overlap
    union_volumes = pred_volumes[:, None] + gt_volumes[None, :] - inter_volumes

    # 3D IoU
    iou_3d = torch.where(union_volumes > 1e-10,
                         inter_volumes / union_volumes,
                         torch.zeros_like(union_volumes))

    return iou_3d


def voc_ap(rec: List[float], prec: List[float]) -> float:
    """Compute VOC 2010 Average Precision."""
    rec = rec.copy()
    prec = prec.copy()
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])

    return ap


def calculate_ap(results: List[dict], iou_thresh: float, use_3d_iou: bool = False,
                 device: str = 'cuda') -> float:
    """Calculate AP for a specific IoU threshold using CUDA acceleration.

    Args:
        results: List of result dicts
        iou_thresh: IoU threshold
        use_3d_iou: Use 3D IoU if True, else BEV IoU
        device: Device to use for computation

    Returns:
        AP value
    """
    all_scores = []
    all_tp = []
    all_fp = []
    total_gt = 0

    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    for result in results:
        pred_bboxes = result['pred_bboxes']
        pred_scores = result['pred_scores']
        pred_labels = result['pred_labels']
        gt_bboxes = result['gt_bboxes']
        gt_labels = result['gt_labels']

        # Filter by class (vehicle = 0)
        pred_mask = pred_labels == 0
        gt_mask = gt_labels == 0

        pred_bboxes_cls = pred_bboxes[pred_mask]
        pred_scores_cls = pred_scores[pred_mask]
        gt_bboxes_cls = gt_bboxes[gt_mask]

        total_gt += len(gt_bboxes_cls)

        if len(pred_bboxes_cls) == 0:
            continue

        if len(gt_bboxes_cls) == 0:
            all_scores.extend(pred_scores_cls.tolist())
            all_fp.extend([1] * len(pred_bboxes_cls))
            all_tp.extend([0] * len(pred_bboxes_cls))
            continue

        # Convert to torch tensors and move to device
        pred_tensor = torch.from_numpy(pred_bboxes_cls[:, :7]).float().to(device)
        gt_tensor = torch.from_numpy(gt_bboxes_cls[:, :7]).float().to(device)

        # Compute BEV IoU matrix (CUDA accelerated)
        bev_iou_matrix = compute_bev_iou_cuda(pred_tensor, gt_tensor)

        if use_3d_iou:
            iou_matrix = compute_3d_iou_cuda(pred_tensor, gt_tensor, bev_iou_matrix)
        else:
            iou_matrix = bev_iou_matrix

        # Move back to CPU for matching
        iou_matrix = iou_matrix.cpu().numpy()

        # Sort predictions by score
        score_order = np.argsort(-pred_scores_cls)
        gt_matched = np.zeros(len(gt_bboxes_cls), dtype=bool)

        for pred_idx in score_order:
            all_scores.append(pred_scores_cls[pred_idx])

            if gt_matched.all():
                all_fp.append(1)
                all_tp.append(0)
                continue

            # Find best unmatched GT
            ious = iou_matrix[pred_idx].copy()
            ious[gt_matched] = -1
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]

            if best_iou >= iou_thresh:
                all_tp.append(1)
                all_fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                all_tp.append(0)
                all_fp.append(1)

    if total_gt == 0 or len(all_tp) == 0:
        return 0.0

    # Sort by score
    sorted_indices = np.argsort(-np.array(all_scores))
    all_tp = np.array(all_tp)[sorted_indices]
    all_fp = np.array(all_fp)[sorted_indices]

    # Cumulative sum
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(all_fp)

    # Precision and recall
    rec = (tp_cumsum / total_gt).tolist()
    prec = (tp_cumsum / (tp_cumsum + fp_cumsum)).tolist()

    return voc_ap(rec, prec)


@METRICS.register_module()
class OPV2VMetric(BaseMetric):
    """OPV2V evaluation metric.

    Computes both 2D BEV AP and 3D AP with CUDA acceleration.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 iou_thresholds: List[float] = [0.3, 0.5, 0.7],
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'OPV2V metric'
        super(OPV2VMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        self.iou_thresholds = iou_thresholds
        self.backend_args = backend_args
        self.metrics = metric if isinstance(metric, list) else [metric]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions."""
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']

            pred_bboxes = pred_3d['bboxes_3d'].tensor.cpu().numpy()
            pred_scores = pred_3d['scores_3d'].cpu().numpy()
            pred_labels = pred_3d['labels_3d'].cpu().numpy()

            result['pred_bboxes'] = pred_bboxes
            result['pred_scores'] = pred_scores
            result['pred_labels'] = pred_labels
            result['sample_idx'] = data_sample['sample_idx']

            if 'eval_ann_info' in data_sample and data_sample['eval_ann_info'] is not None:
                gt_bboxes_raw = data_sample['eval_ann_info']['gt_bboxes_3d']
                gt_labels_raw = data_sample['eval_ann_info']['gt_labels_3d']
            elif 'gt_instances_3d' in data_sample:
                gt_bboxes_raw = data_sample['gt_instances_3d']['bboxes_3d']
                gt_labels_raw = data_sample['gt_instances_3d']['labels_3d']
            else:
                gt_bboxes_raw = None
                gt_labels_raw = None

            if gt_bboxes_raw is None:
                gt_bboxes = np.zeros((0, 7))
                gt_labels = np.zeros(0, dtype=np.int64)
            else:
                if hasattr(gt_bboxes_raw, 'tensor'):
                    gt_bboxes = gt_bboxes_raw.tensor.cpu().numpy()
                elif hasattr(gt_bboxes_raw, 'cpu'):
                    gt_bboxes = gt_bboxes_raw.cpu().numpy()
                else:
                    gt_bboxes = np.array(gt_bboxes_raw)

                if hasattr(gt_labels_raw, 'cpu'):
                    gt_labels = gt_labels_raw.cpu().numpy()
                else:
                    gt_labels = np.array(gt_labels_raw)

            result['gt_bboxes'] = gt_bboxes
            result['gt_labels'] = gt_labels

            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()

        metrics_dict = {}

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print_log('\n' + '='*60, logger=logger)
        print_log(f'OPV2V Evaluation Results (using {device.upper()})', logger=logger)
        print_log('='*60, logger=logger)

        for iou_thresh in self.iou_thresholds:
            # 2D BEV AP
            ap_bev = calculate_ap(results, iou_thresh, use_3d_iou=False, device=device)
            metrics_dict[f'AP_BEV@{iou_thresh}'] = ap_bev

            # 3D AP
            ap_3d = calculate_ap(results, iou_thresh, use_3d_iou=True, device=device)
            metrics_dict[f'AP_3D@{iou_thresh}'] = ap_3d

            print_log(f'\n--- IoU Threshold: {iou_thresh} ---', logger=logger)
            print_log(f'  AP_BEV (2D): {ap_bev:.4f}', logger=logger)
            print_log(f'  AP_3D:       {ap_3d:.4f}', logger=logger)

        # Overall mAP
        mAP_bev = np.mean([metrics_dict[f'AP_BEV@{t}'] for t in self.iou_thresholds])
        mAP_3d = np.mean([metrics_dict[f'AP_3D@{t}'] for t in self.iou_thresholds])

        metrics_dict['mAP_BEV'] = mAP_bev
        metrics_dict['mAP_3D'] = mAP_3d

        print_log(f'\n{"="*60}', logger=logger)
        print_log(f'Summary:', logger=logger)
        print_log(f'  mAP_BEV (2D): {mAP_bev:.4f}', logger=logger)
        print_log(f'  mAP_3D:       {mAP_3d:.4f}', logger=logger)
        print_log(f'{"="*60}\n', logger=logger)

        return metrics_dict
