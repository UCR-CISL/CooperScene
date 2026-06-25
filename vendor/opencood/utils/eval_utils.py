# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
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
    """
    Calculate TP/FP for one frame.

    iou_mode='bev'  -> shapely polygon IoU on xy (original OpenCOOD behavior)
    iou_mode='3d'   -> polygon BEV intersection * z-overlap / 3D union
    """
    assert iou_mode in ('bev', '3d')

    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))
        # 3D mode also needs the raw corner arrays for z range
        gt_box_list = list(gt_boxes) if iou_mode == '3d' else None

        for i in range(score_order_descend.shape[0]):
            idx = score_order_descend[i]
            det_polygon = det_polygon_list[idx]

            if iou_mode == 'bev':
                ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
            else:
                if len(gt_polygon_list) == 0:
                    ious = np.array([], dtype=np.float32)
                else:
                    gt_box_arr = np.stack(gt_box_list, axis=0)
                    ious = common_utils.compute_3d_iou(
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
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
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


def eval_final_results(result_stat, save_path, global_sort_detections,
                       label='bev'):
    """Print + dump AP@0.3/0.5/0.7 for one IoU mode (bev|3d)."""
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })

    suffix = '' if label == 'bev' else f'_{label}'
    name = 'eval' + suffix
    if global_sort_detections:
        name += '_global_sort'
    output_file = name + '.yaml'
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    tag = label.upper()
    print(f'{tag} AP @ IoU 0.3 = %.3f,  @ IoU 0.5 = %.3f,  @ IoU 0.7 = %.3f'
          % (ap_30, ap_50, ap_70))
    return ap_30, ap_50, ap_70
