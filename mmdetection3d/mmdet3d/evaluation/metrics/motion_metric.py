"""Motion Prediction Evaluation Metric.

Computes standard motion prediction metrics:
- minADE_k: Minimum Average Displacement Error over k modes
- minFDE_k: Minimum Final Displacement Error over k modes
- MissRate_k: Rate of predictions with FDE > threshold
- EPA: End-point Average (optional)

Compatible with UniAD-style motion head outputs.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS


@METRICS.register_module()
class MotionMetric(BaseMetric):
    """Motion prediction evaluation metric.

    Args:
        miss_rate_threshold (float): FDE threshold for miss rate. Default: 2.0.
        num_modes_eval (list[int]): Number of modes to evaluate. Default: [1, 6].
        predict_steps (int): Number of prediction steps. Default: 12.
        step_interval (float): Time interval per step in seconds. Default: 0.5.
    """

    def __init__(self,
                 miss_rate_threshold: float = 2.0,
                 num_modes_eval: List[int] = [1, 6],
                 predict_steps: int = 12,
                 step_interval: float = 0.5,
                 eval_horizons: Optional[List[float]] = None,
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 **kwargs) -> None:
        self.default_prefix = 'Motion'
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.miss_rate_threshold = miss_rate_threshold
        self.num_modes_eval = num_modes_eval
        self.predict_steps = predict_steps
        self.step_interval = step_interval
        # Time horizons (in seconds) for ADE/FDE breakdown
        if eval_horizons is None:
            eval_horizons = [1.0, 3.0, 5.0, 6.0]
        # Convert seconds → step indices (0-based)
        self.eval_horizon_steps = {}
        for t in eval_horizons:
            step = int(round(t / step_interval))
            if 1 <= step <= predict_steps:
                self.eval_horizon_steps[t] = step

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of predictions.

        Expects data_samples to have:
        - pred_traj: (A, P, T, 2) or (A, T, 2) predicted trajectories
        - pred_traj_score: (A, P) mode scores
        - metainfo.gt_fut_traj: (N, T, 2) GT future trajectories
        - metainfo.gt_fut_traj_mask: (N, T) validity mask
        """
        for data_sample in data_samples:
            result = dict()

            # Predictions
            if 'pred_traj' in data_sample:
                pred_traj = data_sample['pred_traj']
                if hasattr(pred_traj, 'cpu'):
                    pred_traj = pred_traj.cpu().numpy()

                pred_score = data_sample.get('pred_traj_score', None)
                if pred_score is not None and hasattr(pred_score, 'cpu'):
                    pred_score = pred_score.cpu().numpy()

                result['pred_traj'] = pred_traj  # (A, P, T, 2) or (A, T, 2)
                result['pred_score'] = pred_score  # (A, P) or None
            else:
                result['pred_traj'] = np.zeros((0, 1, self.predict_steps, 2))
                result['pred_score'] = None

            # Ground truth
            meta = data_sample.get('metainfo', data_sample)
            gt_traj = meta.get('gt_fut_traj', np.zeros((0, self.predict_steps, 2)))
            gt_mask = meta.get('gt_fut_traj_mask', np.zeros((0, self.predict_steps)))
            if hasattr(gt_traj, 'cpu'):
                gt_traj = gt_traj.cpu().numpy()
            if hasattr(gt_mask, 'cpu'):
                gt_mask = gt_mask.cpu().numpy()

            result['gt_traj'] = gt_traj    # (N, T, 2)
            result['gt_mask'] = gt_mask    # (N, T)

            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute motion prediction metrics."""
        logger: MMLogger = MMLogger.get_current_instance()

        # Collect per-agent errors
        # {num_modes: {'ade': [...], 'fde': [], 'ade_Ts': {T: [...]}, 'fde_Ts': {T: [...]}}}
        all_errors = {
            k: {
                'ade': [], 'fde': [],
                'ade_Ts': {t: [] for t in self.eval_horizon_steps},
                'fde_Ts': {t: [] for t in self.eval_horizon_steps},
            }
            for k in self.num_modes_eval
        }

        for result in results:
            pred_traj = result['pred_traj']  # (A, P, T, 2) or (A, T, 2)
            pred_score = result['pred_score']
            gt_traj = result['gt_traj']      # (N, T, 2)
            gt_mask = result['gt_mask']       # (N, T)

            if pred_traj.ndim == 3:
                # (A, T, 2) → (A, 1, T, 2), single mode
                pred_traj = pred_traj[:, np.newaxis]

            A, P, T_pred, _ = pred_traj.shape
            N = gt_traj.shape[0]

            if A == 0 or N == 0:
                continue

            # Hungarian matching: use best-mode ADE as cost
            T_match = min(T_pred, gt_traj.shape[1])
            # pred best-mode center: (A, T, 2)
            if pred_score is not None:
                best_mode = np.argmax(pred_score, axis=1)  # (A,)
                pred_best = pred_traj[
                    np.arange(A), best_mode]  # (A, T, 2)
            else:
                pred_best = pred_traj[:, 0]  # (A, T, 2)

            # Cost matrix: ADE between pred_best and gt
            cost = np.zeros((A, N))
            for ai in range(A):
                for ni in range(N):
                    mask_n = gt_mask[ni, :T_match]
                    valid = mask_n.sum()
                    if valid == 0:
                        cost[ai, ni] = 1e6
                    else:
                        err = np.sqrt(((pred_best[ai, :T_match] -
                                        gt_traj[ni, :T_match]) ** 2).sum(-1))
                        cost[ai, ni] = (err * mask_n).sum() / valid

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            for ri, ci in zip(row_ind, col_ind):
                gt_i = gt_traj[ci]     # (T, 2)
                mask_i = gt_mask[ci]   # (T,)
                T_valid = int(mask_i.sum())
                if T_valid == 0:
                    continue

                pred_i = pred_traj[ri]  # (P, T, 2)

                for k in self.num_modes_eval:
                    k_actual = min(k, P)

                    # Select top-k modes by score
                    if pred_score is not None and k_actual < P:
                        scores_i = pred_score[ri]  # (P,)
                        topk_idx = np.argsort(-scores_i)[:k_actual]
                        pred_k = pred_i[topk_idx]  # (k, T, 2)
                    else:
                        pred_k = pred_i[:k_actual]  # (k, T, 2)

                    # Compute per-mode ADE and FDE
                    T_use = min(T_pred, T_valid)
                    errors = np.sqrt(
                        ((pred_k[:, :T_use] - gt_i[np.newaxis, :T_use]) ** 2).sum(-1)
                    )  # (k, T_use)

                    # Mask invalid steps
                    mask_use = mask_i[:T_use]
                    errors = errors * mask_use[np.newaxis]

                    # ADE per mode: mean over valid steps
                    valid_count = mask_use.sum()
                    if valid_count == 0:
                        continue
                    ade_per_mode = errors.sum(axis=1) / valid_count  # (k,)
                    # FDE per mode: error at last valid step
                    last_valid = T_use - 1
                    fde_per_mode = errors[:, last_valid]  # (k,)

                    # minADE, minFDE (full horizon)
                    best_mode_idx = ade_per_mode.argmin()
                    all_errors[k]['ade'].append(ade_per_mode[best_mode_idx])
                    all_errors[k]['fde'].append(fde_per_mode[best_mode_idx])

                    # Per-horizon ADE/FDE using same best mode
                    best_pred = pred_k[best_mode_idx]  # (T, 2)
                    for t_sec, t_step in self.eval_horizon_steps.items():
                        t_end = min(t_step, T_use)
                        mask_h = mask_i[:t_end]
                        valid_h = mask_h.sum()
                        if valid_h == 0:
                            continue
                        err_h = np.sqrt(
                            ((best_pred[:t_end] -
                              gt_i[:t_end]) ** 2).sum(-1))
                        err_h = err_h * mask_h
                        ade_h = err_h.sum() / valid_h
                        fde_h = err_h[t_end - 1]
                        all_errors[k]['ade_Ts'][t_sec].append(ade_h)
                        all_errors[k]['fde_Ts'][t_sec].append(fde_h)

        # Aggregate
        metrics_dict = {}
        print_log('\n' + '=' * 50, logger=logger)
        print_log('Motion Prediction Evaluation', logger=logger)
        print_log('=' * 50, logger=logger)

        for k in self.num_modes_eval:
            ades = all_errors[k]['ade']
            fdes = all_errors[k]['fde']
            if len(ades) == 0:
                continue

            minADE = np.mean(ades)
            minFDE = np.mean(fdes)
            miss_rate = np.mean([f > self.miss_rate_threshold for f in fdes])

            metrics_dict[f'minADE_{k}'] = float(minADE)
            metrics_dict[f'minFDE_{k}'] = float(minFDE)
            metrics_dict[f'MissRate_{k}'] = float(miss_rate)

            print_log(f'  minADE_{k}: {minADE:.4f} m', logger=logger)
            print_log(f'  minFDE_{k}: {minFDE:.4f} m', logger=logger)
            print_log(
                f'  MissRate_{k} (>{self.miss_rate_threshold}m): '
                f'{miss_rate:.4f}', logger=logger)

            # Per-horizon ADE/FDE
            for t_sec in sorted(self.eval_horizon_steps.keys()):
                ades_t = all_errors[k]['ade_Ts'][t_sec]
                fdes_t = all_errors[k]['fde_Ts'][t_sec]
                if len(ades_t) == 0:
                    continue
                ade_t = np.mean(ades_t)
                fde_t = np.mean(fdes_t)
                t_label = f'{t_sec:g}s'
                metrics_dict[f'minADE_{k}@{t_label}'] = float(ade_t)
                metrics_dict[f'minFDE_{k}@{t_label}'] = float(fde_t)
                print_log(
                    f'    @{t_label}: ADE={ade_t:.4f}  FDE={fde_t:.4f}',
                    logger=logger)

        total_agents = len(all_errors[self.num_modes_eval[0]]['ade'])
        print_log(f'\n  Total evaluated agents: {total_agents}', logger=logger)
        print_log('=' * 50 + '\n', logger=logger)

        return metrics_dict
