"""Single-agent BEVFusion + UniAD end-to-end model.

Architecture:
    BEVFusion (LiDAR, single-agent)
        -> BEV features (B, 512, 180, 180)
        -> TransFusionHead detections
            |
    BEV Adapter (512->256, 180->200)
            |
    TrackFormer (temporal enrichment)
        -> track queries with velocity + past trajectory encoding
            |
    MotionHead (MotionFormer decoder, 3 layers)
        -> 12-step trajectory prediction, 6 modes per agent
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModel

logger = logging.getLogger(__name__)


@MODELS.register_module()
class BEVFusionUniAD(BaseModel):
    """Single-agent BEVFusion + UniAD motion prediction pipeline.

    Args:
        bevfusion (dict): Config for BEVFusion perception model.
        bev_adapter (dict): Config for BEV feature adapter.
        motion_head (dict): Config for MotionHead.
        map_encoder (dict): Config for BEVMapEncoder (optional).
        trackformer (dict): Config for TrackFormer (optional).
        planning_head (dict): Config for PlanningHead (optional).
        freeze_perception (bool): Whether to freeze BEVFusion backbone.
        bevfusion_checkpoint (str): Path to pretrained BEVFusion weights.
    """

    def __init__(self,
                 bevfusion=None,
                 bev_adapter=dict(
                     in_channels=512,
                     out_channels=256,
                     in_size=180,
                     out_size=200,
                 ),
                 motion_head=dict(type='MotionHead'),
                 map_encoder=None,
                 trackformer=None,
                 planning_head=None,
                 freeze_perception=True,
                 bevfusion_checkpoint=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.bevfusion_checkpoint = bevfusion_checkpoint

        # BEVFusion perception backbone
        if bevfusion is not None:
            self.bevfusion = MODELS.build(bevfusion)
        else:
            self.bevfusion = None

        # BEV adapter: project + resize BEVFusion features
        in_ch = bev_adapter['in_channels']
        out_ch = bev_adapter['out_channels']
        self.bev_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.in_size = bev_adapter.get('in_size', 180)
        self.out_size = bev_adapter.get('out_size', 200)
        self.out_channels = out_ch

        # TrackFormer: temporal track query enrichment
        if trackformer is not None:
            self.trackformer = MODELS.build(trackformer)
        else:
            self.trackformer = None

        # Motion prediction head
        self.motion_head = MODELS.build(motion_head)

        # Map encoder (optional)
        if map_encoder is not None:
            self.map_encoder = MODELS.build(map_encoder)
        else:
            self.map_encoder = None

        # Planning head (optional)
        if planning_head is not None:
            self.planning_head = MODELS.build(planning_head)
        else:
            self.planning_head = None

        # Freeze perception
        self.freeze_perception = freeze_perception
        if freeze_perception and self.bevfusion is not None:
            self.bevfusion.eval()
            for param in self.bevfusion.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Load pretrained BEVFusion checkpoint if specified."""
        super().init_weights()
        if self.bevfusion_checkpoint and self.bevfusion is not None:
            self._load_bevfusion_checkpoint(self.bevfusion_checkpoint)

    def _load_bevfusion_checkpoint(self, checkpoint_path):
        """Load BEVFusion checkpoint with key mapping and spconv handling."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        model_state = self.bevfusion.state_dict()
        loaded = 0
        skipped = []

        for k, v in state_dict.items():
            if k not in model_state:
                continue
            ms = model_state[k].shape
            if v.shape == ms:
                model_state[k].copy_(v)
                loaded += 1
            elif v.dim() == 5 and ms == (
                    v.shape[1], *v.shape[2:], v.shape[0]):
                # spconv v1->v2 layout permutation
                model_state[k].copy_(
                    v.permute(1, 2, 3, 4, 0).contiguous())
                loaded += 1
            else:
                skipped.append(k)

        if skipped:
            logger.warning(
                f'Skipped {len(skipped)} size-mismatched keys: '
                f'{skipped[:6]}...')
        logger.info(
            f'Loaded {loaded}/{len(model_state)} params into bevfusion '
            f'from {checkpoint_path}')

    def adapt_bev(self, bev_feat):
        """Adapt BEVFusion features: project channels + resize spatial.

        Args:
            bev_feat: (B, C_in, H_in, W_in) e.g. (B, 512, 180, 180)

        Returns:
            (B, C_out, H_out, W_out) e.g. (B, 256, 200, 200)
        """
        bev_feat = self.bev_proj(bev_feat)
        if bev_feat.shape[-1] != self.out_size:
            bev_feat = F.interpolate(
                bev_feat,
                size=(self.out_size, self.out_size),
                mode='bilinear',
                align_corners=False)
        return bev_feat

    def forward(self, inputs=None, data_samples=None, mode='loss', **kwargs):
        """Unified forward pass.

        Args:
            inputs: Raw sensor inputs for BEVFusion.
            data_samples: List of Det3DDataSample.
            mode: 'loss', 'predict', or 'tensor'.
        """
        # Step 1: Run perception
        if self.bevfusion is not None:
            if self.freeze_perception:
                self.bevfusion.eval()
                with torch.no_grad():
                    bev_feat, det_results = self._run_perception(
                        inputs, data_samples)
            else:
                bev_feat, det_results = self._run_perception(
                    inputs, data_samples)
        else:
            raise RuntimeError('BEVFusion backbone is required')

        # Step 2: Adapt BEV features
        bev_adapted = self.adapt_bev(bev_feat)  # (B, 256, 200, 200)

        # Step 3: Generate track queries
        if self.trackformer is not None:
            track_query, track_boxes = self.trackformer(
                bev_adapted, det_results, data_samples)
        else:
            track_query, track_boxes = self._extract_track_queries_simple(
                det_results, bev_adapted)

        # Step 4: Encode map data into lane queries
        lane_query, lane_query_pos = None, None
        if self.map_encoder is not None:
            bev_map = self._extract_bev_map(data_samples)
            if bev_map is not None:
                lane_query, lane_query_pos = self.map_encoder(bev_map)

        # Step 5: Motion prediction
        outs_motion = self.motion_head(
            bev_embed=bev_adapted.detach(),
            track_query=track_query,
            track_boxes=track_boxes,
            lane_query=lane_query,
            lane_query_pos=lane_query_pos,
        )

        # Step 6: Planning (optional)
        outs_planning = None
        if self.planning_head is not None:
            outs_planning = self.planning_head(
                bev_embed=bev_adapted,
                outs_motion=outs_motion,
                command=kwargs.get('command', None),
            )

        if mode == 'loss':
            return self._compute_losses(
                outs_motion, outs_planning, data_samples,
                track_boxes=track_boxes)
        elif mode == 'predict':
            return self._predict(
                outs_motion, outs_planning, det_results, data_samples)
        else:
            return dict(
                bev_feat=bev_adapted,
                outs_motion=outs_motion,
                det_results=det_results,
            )

    def _run_perception(self, inputs, data_samples):
        """Run BEVFusion to get BEV features + detections.

        Args:
            inputs: Dict with 'points' (list of tensors).
            data_samples: List of Det3DDataSample.

        Returns:
            bev_feat: (B, C, H, W) BEV features.
            det_results: List of Det3DDataSample with pred_instances_3d.
        """
        batch_input_metas = [item.metainfo for item in data_samples]

        feats = self.bevfusion.extract_feat(inputs, batch_input_metas)
        bev_feat = feats[0]  # (B, 512, H, W)

        det_results = self.bevfusion.bbox_head.predict(
            feats, batch_input_metas)
        det_samples = self.bevfusion.add_pred_to_datasample(
            data_samples, det_results)

        return bev_feat, det_samples

    def _extract_track_queries_simple(self, det_results, bev_feat):
        """Fallback: extract track queries without temporal enrichment.

        Used when TrackFormer is not configured. Simply samples BEV features
        at detection centers.
        """
        B, D, H, W = bev_feat.shape
        device = bev_feat.device
        pc_range = self.motion_head.pc_range

        all_queries = []
        all_boxes = []

        for b in range(B):
            if hasattr(det_results[b], 'pred_instances_3d'):
                pred = det_results[b].pred_instances_3d
                boxes = pred.bboxes_3d.tensor
                scores = pred.scores_3d

                max_agents = 20
                if len(scores) > max_agents:
                    topk_idx = scores.topk(max_agents)[1]
                    boxes = boxes[topk_idx]

                N = boxes.shape[0]
                if N > 0:
                    cx, cy = boxes[:, 0], boxes[:, 1]
                    grid_x = (2.0 * (cx - pc_range[0])
                              / (pc_range[3] - pc_range[0]) - 1.0)
                    grid_y = (2.0 * (cy - pc_range[1])
                              / (pc_range[4] - pc_range[1]) - 1.0)
                    grid = torch.stack([grid_x, grid_y], dim=-1)
                    grid = grid.view(1, N, 1, 2)
                    query = F.grid_sample(
                        bev_feat[b:b + 1], grid,
                        mode='bilinear', padding_mode='zeros',
                        align_corners=False)
                    query = query.squeeze(-1).squeeze(0).permute(1, 0)

                    boxes_10 = torch.zeros(N, 10, device=device)
                    boxes_10[:, 0] = boxes[:, 0]
                    boxes_10[:, 1] = boxes[:, 1]
                    boxes_10[:, 2] = boxes[:, 3]
                    boxes_10[:, 3] = boxes[:, 4]
                    boxes_10[:, 4] = boxes[:, 2]
                    boxes_10[:, 5] = boxes[:, 5]
                    boxes_10[:, 6] = torch.sin(boxes[:, 6])
                    boxes_10[:, 7] = torch.cos(boxes[:, 6])

                    all_queries.append(query)
                    all_boxes.append(boxes_10)
                else:
                    all_queries.append(torch.zeros(0, D, device=device))
                    all_boxes.append(torch.zeros(0, 10, device=device))
            else:
                all_queries.append(torch.zeros(0, D, device=device))
                all_boxes.append(torch.zeros(0, 10, device=device))

        max_a = max(q.shape[0] for q in all_queries)
        max_a = max(max_a, 1)
        track_query = torch.zeros(B, max_a, D, device=device)
        track_boxes = torch.zeros(B, max_a, 10, device=device)
        for b in range(B):
            N = all_queries[b].shape[0]
            if N > 0:
                track_query[b, :N] = all_queries[b]
                track_boxes[b, :N] = all_boxes[b]

        return track_query, track_boxes

    def _extract_bev_map(self, data_samples):
        """Extract BEV map tensor from data_samples metainfo."""
        if data_samples is None:
            return None
        device = next(self.parameters()).device
        maps = []
        for ds in data_samples:
            bm = ds.metainfo.get('bev_map', None)
            if bm is not None:
                maps.append(torch.as_tensor(bm, dtype=torch.float32))
            else:
                return None
        return torch.stack(maps, dim=0).to(device)

    def _extract_trajectory_gt(self, data_samples):
        """Extract trajectory GT from data_samples metainfo.

        Returns:
            gt_fut_traj: (B, A_max, T, 2) padded future trajectories.
            gt_fut_traj_mask: (B, A_max, T) validity mask.
        """
        device = next(self.parameters()).device
        B = len(data_samples)

        fut_trajs = []
        fut_masks = []
        for ds in data_samples:
            meta = ds.metainfo
            ft = meta.get('gt_fut_traj', None)
            fm = meta.get('gt_fut_traj_mask', None)
            if ft is not None:
                fut_trajs.append(
                    torch.as_tensor(ft, dtype=torch.float32, device=device))
                fut_masks.append(
                    torch.as_tensor(fm, dtype=torch.float32, device=device))
            else:
                fut_trajs.append(torch.zeros(0, 12, 2, device=device))
                fut_masks.append(torch.zeros(0, 12, device=device))

        max_a = max(t.shape[0] for t in fut_trajs)
        max_a = max(max_a, 1)
        T = fut_trajs[0].shape[1] if fut_trajs[0].shape[0] > 0 else 12

        gt_fut_traj = torch.zeros(B, max_a, T, 2, device=device)
        gt_fut_traj_mask = torch.zeros(B, max_a, T, device=device)

        for b in range(B):
            n = fut_trajs[b].shape[0]
            if n > 0:
                gt_fut_traj[b, :n] = fut_trajs[b]
                gt_fut_traj_mask[b, :n] = fut_masks[b]

        return gt_fut_traj, gt_fut_traj_mask

    def _compute_losses(self, outs_motion, outs_planning, data_samples,
                        track_boxes=None):
        """Compute all losses."""
        losses = dict()

        gt_fut_traj, gt_fut_traj_mask = self._extract_trajectory_gt(
            data_samples)

        gt_bboxes_3d = []
        for ds in data_samples:
            if (hasattr(ds, 'gt_instances_3d')
                    and 'bboxes_3d' in ds.gt_instances_3d):
                gt_bboxes_3d.append(ds.gt_instances_3d.bboxes_3d)
            else:
                gt_bboxes_3d.append(None)

        if gt_fut_traj.shape[1] > 0:
            motion_losses = self.motion_head.loss(
                outs_motion, gt_fut_traj, gt_fut_traj_mask,
                track_boxes=track_boxes, gt_bboxes_3d=gt_bboxes_3d)
            losses.update(motion_losses)

        if outs_planning is not None and self.planning_head is not None:
            sdc_planning = None  # TODO: add ego planning GT
            if sdc_planning is not None:
                plan_losses = self.planning_head.loss(
                    outs_planning, sdc_planning, None)
                losses.update(plan_losses)

        return losses

    def _predict(self, outs_motion, outs_planning, det_results, data_samples):
        """Generate predictions for inference."""
        B = outs_motion['all_traj_preds'].shape[1]

        for b in range(B):
            traj_scores = outs_motion['all_traj_scores'][-1, b]  # (A, P)
            traj_preds = outs_motion['all_traj_preds'][-1, b]    # (A, P, T, 5)
            best_mode = traj_scores.argmax(dim=-1)
            best_traj = traj_preds[
                torch.arange(len(best_mode)), best_mode]

            det_results[b].pred_traj = best_traj[..., :2]
            det_results[b].pred_traj_scores = traj_scores
            det_results[b].all_traj_preds = traj_preds

        return det_results
