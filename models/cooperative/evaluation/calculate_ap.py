"""Self-contained AP / IoU utilities for cooperative evaluation.

BEV polygon IoU + 3D IoU matching and VOC-style AP, with no external
dependencies beyond numpy / torch / shapely. Used by EvalMetric so the metric
runs in any environment.
"""

import numpy as np
import torch
from shapely.geometry import Polygon


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def torch_tensor_to_numpy(torch_tensor):
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def rotate_points_along_z(points, angle):
    """points: (B, N, 3 + C); angle: (B) radians about z (x ==> y)."""
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def convert_format(boxes_array):
    """(N,4,2) or (N,8,3) -> list[shapely Polygon] from the first 4 xy corners."""
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)


def compute_iou(box, boxes):
    """BEV polygon IoU between one Polygon and a list of Polygons."""
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]
    return np.array(iou, dtype=np.float32)


def compute_3d_iou(det_polygon, gt_polygons, det_box, gt_boxes):
    """3D IoU = BEV polygon intersection * z overlap / 3D union (upright boxes).

    det_box: (8,3) corners; gt_boxes: (M,8,3) corners.
    """
    det_z_min, det_z_max = float(det_box[:, 2].min()), float(det_box[:, 2].max())
    det_h = det_z_max - det_z_min
    det_vol = det_polygon.area * det_h

    ious = []
    for i, gt_poly in enumerate(gt_polygons):
        gz_min = float(gt_boxes[i, :, 2].min())
        gz_max = float(gt_boxes[i, :, 2].max())
        gt_h = gz_max - gz_min
        gt_vol = gt_poly.area * gt_h

        bev_inter = det_polygon.intersection(gt_poly).area
        z_overlap = max(0.0, min(det_z_max, gz_max) - max(det_z_min, gz_min))
        inter_3d = bev_inter * z_overlap
        union_3d = det_vol + gt_vol - inter_3d
        ious.append(float(inter_3d / union_3d) if union_3d > 0 else 0.0)
    return np.array(ious, dtype=np.float32)


def boxes_to_corners_3d(boxes3d, order):
    """(N,7) [x,y,z,dx,dy,dz,heading] -> (N,8,3) corners. order: 'lwh'|'hwl'."""
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d_.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3),
                                      boxes3d_[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def voc_ap(rec, prec):
    """VOC 2010 Average Precision."""
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
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    iou_mode='bev'):
    """TP/FP for one frame. iou_mode='bev' (xy polygon) or '3d' (bev * z)."""
    assert iou_mode in ('bev', '3d')

    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        det_boxes = torch_tensor_to_numpy(det_boxes)
        det_score = torch_tensor_to_numpy(det_score)
        gt_boxes = torch_tensor_to_numpy(gt_boxes)

        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]
        det_polygon_list = list(convert_format(det_boxes))
        gt_polygon_list = list(convert_format(gt_boxes))
        gt_box_list = list(gt_boxes) if iou_mode == '3d' else None

        for i in range(score_order_descend.shape[0]):
            idx = score_order_descend[i]
            det_polygon = det_polygon_list[idx]

            if iou_mode == 'bev':
                ious = compute_iou(det_polygon, gt_polygon_list)
            else:
                if len(gt_polygon_list) == 0:
                    ious = np.array([], dtype=np.float32)
                else:
                    gt_box_arr = np.stack(gt_box_list, axis=0)
                    ious = compute_3d_iou(
                        det_polygon, gt_polygon_list,
                        det_boxes[idx], gt_box_arr)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = int(np.argmax(ious))
            gt_polygon_list.pop(gt_index)
            if gt_box_list is not None:
                gt_box_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou, global_sort_detections):
    """Average precision/recall for a given IoU threshold."""
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec
