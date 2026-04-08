"""MotionFormer transformer modules ported from UniAD as pure PyTorch.

Original: UniAD/projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/modules.py

Uses nn.TransformerEncoderLayer / nn.TransformerDecoderLayer to match the
original UniAD architecture (which also uses these standard PyTorch layers).
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionDeformableAttention(nn.Module):
    """Deformable attention for motion prediction.

    Each motion query attends to BEV features at deformable sampling locations.
    Ported from UniAD's MotionDeformableAttention.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32)
        thetas = thetas * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, value, reference_points,
                spatial_shapes, level_start_index):
        """
        Args:
            query: (B*A*P, 1, D)
            value: (B, H*W, D)
            reference_points: (B*A*P, 1, num_levels, 2) normalized [0,1]
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
        Returns:
            (B*A*P, 1, D)
        """
        bs_query, num_query, _ = query.shape
        bs_value, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs_value, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs_query, num_query, self.num_heads,
            self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            bs_query, num_query, self.num_heads,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs_query, num_query, self.num_heads,
            self.num_levels, self.num_points)

        # Compute sampling locations
        offset_normalizer = spatial_shapes.flip(-1)[None, None, None, :, None, :]
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer
        )

        # Bilinear sampling from value
        output = self._bilinear_sample(
            value, sampling_locations, attention_weights, spatial_shapes)
        output = self.output_proj(output)
        return self.dropout(output) + query

    def _bilinear_sample(self, value, sampling_locations, attention_weights,
                         spatial_shapes):
        """PyTorch implementation of multi-scale deformable attention sampling."""
        bs, num_value, num_heads, head_dim = value.shape
        bs_query = sampling_locations.shape[0]

        # For single-level case (typical in motion prediction)
        if self.num_levels == 1:
            H, W = spatial_shapes[0]
            value_l = value.view(bs, H, W, num_heads, head_dim)
            value_l = value_l.permute(0, 3, 4, 1, 2)  # (B, nH, D, H, W)

            # Expand value for batched queries
            if bs_query != bs:
                ratio = bs_query // bs
                value_l = value_l.unsqueeze(1).expand(-1, ratio, -1, -1, -1, -1)
                value_l = value_l.reshape(bs_query, num_heads, head_dim, H, W)

            locs = sampling_locations[:, :, :, 0, :, :]  # (bs_q, Q, nH, P, 2)
            grid = 2.0 * locs - 1.0  # normalize to [-1, 1]
            # (bs_q, nH, Q*P, 2)
            nQ, nP = locs.shape[1], locs.shape[3]
            grid = grid.permute(0, 2, 1, 3, 4).reshape(bs_query, num_heads, nQ * nP, 2)

            sampled = F.grid_sample(
                value_l.flatten(0, 1),  # (bs_q*nH, D, H, W)
                grid.flatten(0, 1).unsqueeze(1),  # (bs_q*nH, 1, Q*P, 2)
                mode='bilinear', padding_mode='zeros', align_corners=False
            )  # (bs_q*nH, D, 1, Q*P)
            sampled = sampled.view(bs_query, num_heads, head_dim, nQ, nP)
            sampled = sampled.permute(0, 3, 1, 4, 2)  # (bs_q, Q, nH, P, D)

            weights = attention_weights[:, :, :, 0, :]  # (bs_q, Q, nH, P)
            output = (sampled * weights.unsqueeze(-1)).sum(dim=3)  # (bs_q, Q, nH, D)
            output = output.reshape(bs_query, nQ, num_heads * head_dim)
            return output
        else:
            raise NotImplementedError("Multi-level deformable attention not yet ported")


class IntentionInteraction(nn.Module):
    """Self-attention among K-means anchor modes per agent.

    Uses nn.TransformerEncoderLayer matching the original UniAD implementation.
    Input shape: (B, A, P, D) or (B*A, P, D).
    """

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.interaction_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=True,
        )

    def forward(self, query):
        """
        Args:
            query: (B, A, P, D) - per-agent mode queries
        Returns:
            (B, A, P, D)
        """
        B, A, P, D = query.shape
        # B, A, P, D -> B*A, P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out


class TrackAgentInteraction(nn.Module):
    """Cross-attention between motion queries and track queries.

    Uses nn.TransformerDecoderLayer matching the original UniAD implementation.
    """

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=True,
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        Args:
            query: (B, A, P, D) - context query
            key: (B, A, D) - track queries
            query_pos: (B, A, P, D) - mode positional embedding (optional)
            key_pos: (B, A, D) - track query positional embedding (optional)
        Returns:
            (B, A, P, D)
        """
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        # key: (B, A, D) -> (B*A, 1, D) each agent gets its own track query
        mem = key.reshape(B * A, 1, D)
        # query: (B, A, P, D) -> (B*A, P, D)
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class MapInteraction(nn.Module):
    """Cross-attention between motion queries and lane/map queries.

    Uses nn.TransformerDecoderLayer matching the original UniAD implementation.
    """

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=True,
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        Args:
            query: (B, A, P, D) - context query
            key: (B, M, D) - lane queries
            query_pos: (B, A, P, D) - mode positional embedding (optional)
            key_pos: (B, M, D) - lane query positional embedding (optional)
        Returns:
            (B, A, P, D)
        """
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        # query: (B, A, P, D) -> (B*A, P, D)
        query = torch.flatten(query, start_dim=0, end_dim=1)
        # key: (B, M, D) -> (B*A, M, D) broadcast across agents
        mem = key.unsqueeze(1).expand(B, A, -1, -1).reshape(B * A, -1, D)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class MotionTransformerDecoderLayer(nn.Module):
    """One layer of MotionFormer decoder.

    Matches the original UniAD structure:
    1. intention_interaction (TransformerEncoderLayer) - self-attn among modes
    2. track_agent_interaction (TransformerDecoderLayer) - cross-attn with track queries
    3. map_interaction (TransformerDecoderLayer) - cross-attn with lane queries
    4. bev_interaction (MotionDeformableAttention) - deformable attn on BEV

    No standalone FFN step since TransformerEncoderLayer/DecoderLayer
    already include FFN internally.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_points=4,
                 dropout=0.1,
                 use_bev_interaction=True):
        super().__init__()
        self.track_agent_interaction = TrackAgentInteraction(
            embed_dims, num_heads, dropout)
        self.map_interaction = MapInteraction(
            embed_dims, num_heads, dropout)

        self.use_bev_interaction = use_bev_interaction
        if use_bev_interaction:
            self.bev_interaction = MotionDeformableAttention(
                embed_dims, num_heads, num_levels=1,
                num_points=num_points, dropout=dropout)

    def forward(self, query, track_query, lane_query, lane_query_pos,
                bev_embed, reference_points, spatial_shapes,
                level_start_index, track_query_pos=None,
                query_pos=None):
        """
        Args:
            query: (B, A, P, D) - motion queries
            track_query: (B, A, D) - track queries
            lane_query: (B, M, D) or None
            lane_query_pos: (B, M, D) or None
            bev_embed: (B, H*W, D) - BEV features
            reference_points: (B, A, P, T, 2) - normalized reference trajs
            spatial_shapes: (1, 2) for single BEV level
            level_start_index: (1,)
            track_query_pos: (B, A, D) or None - track positional embedding
            query_pos: (B, A, P, D) or None - query positional embedding
        Returns:
            (B, A, P, D)
        """
        B, A, P, D = query.shape

        # 1. Track agent interaction (cross-attn with track queries)
        track_query_embed = self.track_agent_interaction(
            query, track_query,
            query_pos=query_pos,
            key_pos=track_query_pos)

        # 2. Map interaction (cross-attn with lane queries)
        if lane_query is not None:
            map_query_embed = self.map_interaction(
                query, lane_query,
                query_pos=query_pos,
                key_pos=lane_query_pos)
        else:
            map_query_embed = torch.zeros_like(query)

        # 3. BEV deformable interaction
        if self.use_bev_interaction and bev_embed is not None:
            # reference_points: (B, A, P, T, 2) -> use last step for attention
            ref_pts = reference_points[:, :, :, -1:, :]  # (B, A, P, 1, 2)
            ref_pts = ref_pts.reshape(B * A * P, 1, 1, 2)  # (B*A*P, 1, 1, 2)

            query_flat = query.view(B * A * P, 1, D)
            if query_pos is not None:
                query_flat = query_flat + query_pos.reshape(B * A * P, 1, D)
            # Pass original (B, H*W, D) bev_embed; _bilinear_sample
            # handles B->B*A*P expansion via expand (no memory copy).
            bev_query_embed = self.bev_interaction(
                query_flat, bev_embed,
                ref_pts, spatial_shapes, level_start_index)
            bev_query_embed = bev_query_embed.view(B, A, P, D)
        else:
            bev_query_embed = torch.zeros_like(query)

        return track_query_embed, map_query_embed, bev_query_embed


class MotionTransformerDecoder(nn.Module):
    """Multi-layer MotionFormer decoder.

    Matches the original UniAD MotionTransformerDecoder structure:
    - IntentionInteraction is applied once before the layer loop (shared)
    - Per-layer: TrackAgentInteraction, MapInteraction, BEV interaction
    - Embedding fusion MLPs to combine outputs
    - Reference point update with cumsum trick for velocity -> position
    """

    def __init__(self,
                 num_layers=3,
                 embed_dims=256,
                 num_heads=8,
                 feedforward_dims=512,
                 num_points=4,
                 dropout=0.1,
                 use_bev_interaction=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        # Intention interaction (applied once, shared across layers)
        self.intention_interaction_layers = IntentionInteraction(
            embed_dims, num_heads, dropout)

        # Per-layer interaction modules
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction(embed_dims, num_heads, dropout)
             for _ in range(num_layers)])
        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction(embed_dims, num_heads, dropout)
             for _ in range(num_layers)])

        self.use_bev_interaction = use_bev_interaction
        if use_bev_interaction:
            self.bev_interaction_layers = nn.ModuleList(
                [MotionDeformableAttention(
                    embed_dims, num_heads, num_levels=1,
                    num_points=num_points, dropout=dropout)
                 for _ in range(num_layers)])

        # Embedding fusion MLPs (matching original UniAD)
        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(embed_dims * 4, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )

    def forward(self, query, track_query, lane_query, lane_query_pos,
                bev_embed, reference_points, spatial_shapes,
                level_start_index, num_agents, num_modes,
                reg_branches=None,
                track_query_pos=None,
                agent_level_embedding=None,
                scene_level_ego_embedding=None,
                scene_level_offset_embedding=None,
                learnable_embed=None,
                agent_level_embedding_layer=None,
                scene_level_ego_embedding_layer=None,
                scene_level_offset_embedding_layer=None,
                track_bbox_results=None):
        """Forward function for MotionTransformerDecoder.

        Args:
            query: (B, A*P, D) - initial motion queries
            track_query: (B, A, D) - track queries
            lane_query: (B, M, D) or None - lane queries
            lane_query_pos: (B, M, D) or None - lane positional embeddings
            bev_embed: (B, H*W, D) - BEV features
            reference_points: (B, A, P, T, 2) - normalized reference trajectories
            spatial_shapes: (1, 2) for single BEV level
            level_start_index: (1,)
            num_agents: A
            num_modes: P
            reg_branches: list of nn.Module or None - per-layer regression heads
            track_query_pos: (B, A, D) or None
            agent_level_embedding: (B, A, P, D) or None
            scene_level_ego_embedding: (B, A, P, D) or None
            scene_level_offset_embedding: (B, A, P, D) or None
            learnable_embed: (B, A, P, D) or None
            agent_level_embedding_layer: nn.Module or None - for updating embeddings
            scene_level_ego_embedding_layer: nn.Module or None
            scene_level_offset_embedding_layer: nn.Module or None
            track_bbox_results: tracking bbox results for coordinate transforms

        Returns:
            intermediate: list of (B, A*P, D) per layer
            intermediate_ref: list of (B, A, P, T, 2) updated reference points
        """
        B = query.shape[0]
        A, P = num_agents, num_modes
        D = self.embed_dims

        intermediate = []
        intermediate_ref = []

        # Prepare track_query broadcast for fusion: (B, A, D) -> (B, A, P, D)
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)
        if track_query_pos is not None:
            track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)
        else:
            track_query_pos_bc = None

        # --- Intention interaction (applied once before layer loop) ---
        if agent_level_embedding is not None:
            agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
            static_intention_embed = agent_level_embedding
            if scene_level_offset_embedding is not None:
                static_intention_embed = static_intention_embed + scene_level_offset_embedding
            if learnable_embed is not None:
                static_intention_embed = static_intention_embed + learnable_embed
        else:
            # Fallback: reshape query to (B, A, P, D) and apply intention interaction
            query_4d = query.view(B, A, P, D)
            query_4d = self.intention_interaction_layers(query_4d)
            static_intention_embed = query_4d
            agent_level_embedding = query_4d
            scene_level_ego_embedding = torch.zeros(B, A, P, D, device=query.device)
            scene_level_offset_embedding = torch.zeros(B, A, P, D, device=query.device)

        reference_trajs_input = reference_points.unsqueeze(4).detach()

        query_embed = torch.zeros(B, A, P, D, device=query.device)
        for lid in range(self.num_layers):
            # --- Fuse static and dynamic intention embedding ---
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding,
                 scene_level_ego_embedding], dim=-1))

            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)

            query_embed = self.in_query_fuser(torch.cat(
                [query_embed, query_embed_intention], dim=-1))

            # --- Track agent interaction ---
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query,
                query_pos=track_query_pos_bc,
                key_pos=track_query_pos)

            # --- Map interaction ---
            if lane_query is not None:
                map_query_embed = self.map_interaction_layers[lid](
                    query_embed, lane_query,
                    query_pos=track_query_pos_bc,
                    key_pos=lane_query_pos)
            else:
                map_query_embed = torch.zeros_like(query_embed)

            # --- BEV deformable interaction ---
            if self.use_bev_interaction and bev_embed is not None:
                ref_pts = reference_trajs_input[:, :, :, -1:, :, :]  # (B, A, P, 1, 1, 2)
                ref_pts = ref_pts.reshape(B * A * P, 1, 1, 2)

                query_flat = query_embed.reshape(B * A * P, 1, D)
                if track_query_pos_bc is not None:
                    query_flat = query_flat + track_query_pos_bc.reshape(B * A * P, 1, D)
                bev_query_embed = self.bev_interaction_layers[lid](
                    query_flat, bev_embed,
                    ref_pts, spatial_shapes, level_start_index)
                bev_query_embed = bev_query_embed.view(B, A, P, D)
            else:
                bev_query_embed = torch.zeros_like(query_embed)

            # --- Fuse outputs from different interaction layers ---
            fuse_input = [track_query_embed, map_query_embed, bev_query_embed,
                          track_query_bc + (track_query_pos_bc if track_query_pos_bc is not None
                                            else torch.zeros_like(track_query_bc))]
            query_embed = torch.cat(fuse_input, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            # --- Update reference points with regression ---
            if reg_branches is not None:
                tmp = reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps = reference_points.shape[:4]
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)

                # cumsum trick: predict velocities, cumsum to get positions
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = tmp[..., :2]
                reference_points = new_reference_trajs.detach()
                reference_trajs_input = reference_points.unsqueeze(4)

                # Update embeddings for next layer if embedding layers are provided
                if (agent_level_embedding_layer is not None and
                        scene_level_ego_embedding_layer is not None and
                        scene_level_offset_embedding_layer is not None):
                    from .functional import norm_points, pos2posemb2d, anchor_coordinate_transform
                    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

                    ep_offset_embed = reference_points.detach()
                    agent_level_embedding = agent_level_embedding_layer(
                        pos2posemb2d(norm_points(ep_offset_embed[..., -1, :], pc_range)))
                    scene_level_ego_embedding = scene_level_ego_embedding_layer(
                        pos2posemb2d(norm_points(ep_offset_embed[..., -1, :], pc_range)))
                    scene_level_offset_embedding = scene_level_offset_embedding_layer(
                        pos2posemb2d(norm_points(ep_offset_embed[..., -1, :], pc_range)))

            intermediate.append(query_embed.view(B, A * P, D))
            intermediate_ref.append(reference_points)

        return intermediate, intermediate_ref
