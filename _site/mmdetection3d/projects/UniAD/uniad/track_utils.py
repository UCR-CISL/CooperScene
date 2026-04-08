"""Track instance management utilities ported from UniAD.

Contains:
- Instances: container for per-track attributes
- RuntimeTrackerBase: track ID assignment
- QueryInteractionModule (QIM): updates track embeddings
- MemoryBank: temporal attention over track history
"""

import copy
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Instances Container =====================

class Instances:
    """Container for instance-level data (ported from UniAD).

    Stores per-track tensors/lists with consistent length.
    Supports boolean/integer indexing and concatenation.
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == '_fields' or name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no field '{name}'")

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, \
                f'Length mismatch: existing {len(self)}, new {data_len}'
        self._fields[name] = value

    def get(self, name: str) -> Any:
        return self._fields[name]

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def __getitem__(self, item) -> 'Instances':
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                ret.set(k, v[item])
            elif isinstance(v, list):
                if isinstance(item, torch.BoolTensor):
                    item_list = item.tolist()
                    ret.set(k, [v[i] for i, b in enumerate(item_list) if b])
                elif isinstance(item, (int, slice)):
                    if isinstance(item, int):
                        ret.set(k, [v[item]])
                    else:
                        ret.set(k, v[item])
                else:
                    ret.set(k, [v[i] for i in item.tolist()])
            else:
                ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError('Empty Instances has no len')

    def __repr__(self) -> str:
        s = f'Instances(num_instances={len(self)}, '
        s += f'image_size={self._image_size}, '
        s += f'fields=[{", ".join(self._fields.keys())}])'
        return s

    @staticmethod
    def cat(instance_lists: List['Instances']) -> 'Instances':
        assert len(instance_lists) > 0
        ret = Instances(instance_lists[0]._image_size)
        all_keys = set()
        for inst in instance_lists:
            all_keys.update(inst._fields.keys())
        for k in all_keys:
            values = []
            for inst in instance_lists:
                if inst.has(k):
                    values.append(inst.get(k))
            if isinstance(values[0], torch.Tensor):
                ret.set(k, torch.cat(values, dim=0))
            elif isinstance(values[0], list):
                ret.set(k, list(itertools.chain.from_iterable(values)))
            else:
                raise ValueError(f'Cannot cat field {k} of type {type(values[0])}')
        return ret

    def to(self, *args, **kwargs) -> 'Instances':
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, 'to'):
                ret.set(k, v.to(*args, **kwargs))
            else:
                ret.set(k, v)
        return ret


# ===================== Runtime Tracker =====================

class RuntimeTrackerBase:
    """Assigns and manages track IDs at runtime (ported from UniAD)."""

    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4,
                 miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances) -> None:
        # Reset disappear time for high-score tracks
        track_instances.disappear_time[
            track_instances.scores >= self.score_thresh] = 0

        for i in range(len(track_instances)):
            # New detection with high score → assign ID
            if (track_instances.obj_idxes[i] == -1 and
                    track_instances.scores[i] >= self.score_thresh):
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1

            # Existing track with low score → increment disappear time
            elif (track_instances.obj_idxes[i] >= 0 and
                  track_instances.scores[i] < self.filter_score_thresh):
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1  # kill track


# ===================== Memory Bank =====================

class MemoryBank(nn.Module):
    """Temporal memory bank with cross-attention (ported from UniAD).

    Stores history of track embeddings and enriches current embeddings
    via temporal cross-attention.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        feedforward_dims (int): FFN hidden dimension.
        memory_bank_len (int): Max history frames to store.
        memory_bank_score_thresh (float): Min score for saving to bank.
        save_period (int): Interval between memory saves.
    """

    def __init__(self, embed_dims=256, num_heads=8, feedforward_dims=2048,
                 memory_bank_len=4, memory_bank_score_thresh=0.0,
                 save_period=3):
        super().__init__()
        self.embed_dims = embed_dims
        self.memory_bank_len = memory_bank_len
        self.save_thresh = memory_bank_score_thresh
        self.save_period = save_period

        self.save_proj = nn.Linear(embed_dims, embed_dims)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=0.0)
        self.temporal_fc1 = nn.Linear(embed_dims, feedforward_dims)
        self.temporal_fc2 = nn.Linear(feedforward_dims, embed_dims)
        self.temporal_norm1 = nn.LayerNorm(embed_dims)
        self.temporal_norm2 = nn.LayerNorm(embed_dims)

    def update(self, track_instances: Instances) -> None:
        """Save current embeddings to memory bank.

        Bank stores detached history — gradients flow through temporal
        cross-attention weights, not through stored bank content.
        """
        # Detach before storing: bank is historical, not part of forward graph
        embed = self.save_proj(track_instances.output_embedding).detach()[:, None]
        scores = track_instances.scores

        if self.training:
            saved_idxes = scores > 0
        else:
            save_period = track_instances.save_period.clone()
            track_instances.save_period += 1
            track_instances.save_period[save_period >= self.save_period] = 0
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)

        prev_mem = track_instances.mem_bank[saved_idxes]
        save_embed = embed[saved_idxes]
        if len(prev_mem) > 0:
            new_mem = torch.cat([prev_mem[:, 1:], save_embed], dim=1)
            # Clone to avoid in-place modification of tensor used in backward
            new_mem_bank = track_instances.mem_bank.clone()
            new_mem_bank[saved_idxes] = new_mem
            track_instances.mem_bank = new_mem_bank
            # Update padding mask
            new_mask = track_instances.mem_padding_mask.clone()
            new_mask[saved_idxes] = torch.cat([
                track_instances.mem_padding_mask[saved_idxes, 1:],
                torch.zeros(len(new_mem), 1, dtype=torch.bool,
                            device=new_mem.device)
            ], dim=1)
            track_instances.mem_padding_mask = new_mask

    def _forward_temporal_attn(self, track_instances: Instances) -> Instances:
        """Apply temporal cross-attention using memory bank."""
        key_padding_mask = track_instances.mem_padding_mask
        # Valid tracks: last position is not masked
        valid_idxes = key_padding_mask[:, -1] == 0

        if valid_idxes.sum() == 0:
            return track_instances

        embed = track_instances.output_embedding[valid_idxes]  # (n, D)
        prev_embed = track_instances.mem_bank[valid_idxes]     # (n, T, D)

        # Temporal cross-attention
        embed2 = self.temporal_attn(
            embed[None],                           # (1, n, D)
            prev_embed.transpose(0, 1),            # (T, n, D)
            prev_embed.transpose(0, 1),            # (T, n, D)
            key_padding_mask=key_padding_mask[valid_idxes],
        )[0][0]  # (n, D)

        embed = self.temporal_norm1(embed + embed2)
        embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
        embed = self.temporal_norm2(embed + embed2)

        # Avoid in-place modification for autograd safety
        new_output_embedding = track_instances.output_embedding.clone()
        new_output_embedding[valid_idxes] = embed
        track_instances.output_embedding = new_output_embedding
        return track_instances

    def forward(self, track_instances: Instances,
                update_bank: bool = True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances


# ===================== Query Interaction Module =====================

class QueryInteractionModule(nn.Module):
    """Query Interaction Module (QIM) ported from UniAD.

    Updates track query embeddings using self-attention and FFN.
    Selects active tracks, applies random dropout (training),
    and merges with newly initialized queries.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        feedforward_dims (int): FFN hidden dimension.
        dropout (float): Dropout rate.
        random_drop (float): Probability of dropping active tracks.
        fp_ratio (float): False positive ratio.
        update_query_pos (bool): Whether to update query position part.
    """

    def __init__(self, embed_dims=256, num_heads=8, feedforward_dims=2048,
                 dropout=0.0, random_drop=0.0, fp_ratio=0.3,
                 update_query_pos=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.random_drop = random_drop
        self.fp_ratio = fp_ratio
        self.update_query_pos = update_query_pos

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout)

        # FFN for output_embedding update
        self.linear1 = nn.Linear(embed_dims, feedforward_dims)
        self.linear2 = nn.Linear(feedforward_dims, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Feature update
        self.linear_feat1 = nn.Linear(embed_dims, feedforward_dims)
        self.linear_feat2 = nn.Linear(feedforward_dims, embed_dims)
        self.norm_feat = nn.LayerNorm(embed_dims)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)

        if update_query_pos:
            self.linear_pos1 = nn.Linear(embed_dims, feedforward_dims)
            self.linear_pos2 = nn.Linear(feedforward_dims, embed_dims)
            self.norm_pos = nn.LayerNorm(embed_dims)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)

    def _select_active_tracks(self, track_instances: Instances,
                               ) -> Instances:
        """Select active tracks for updating."""
        if self.training:
            active_mask = (track_instances.obj_idxes >= 0)
            if hasattr(track_instances, 'iou'):
                active_mask = active_mask & (track_instances.iou > 0.5)

            # Random dropout of active tracks
            if self.random_drop > 0 and active_mask.sum() > 0:
                drop_mask = torch.rand(active_mask.sum(),
                                       device=active_mask.device) < self.random_drop
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                active_mask[active_indices[drop_mask]] = False

            active_instances = track_instances[active_mask]

            # Add false positives from inactive tracks
            inactive_mask = ~active_mask
            if inactive_mask.sum() > 0 and self.fp_ratio > 0:
                num_fp = max(1, int(active_mask.sum() * self.fp_ratio))
                inactive_instances = track_instances[inactive_mask]
                if len(inactive_instances) > 0:
                    fp_scores = inactive_instances.scores
                    _, fp_idx = fp_scores.topk(
                        min(num_fp, len(inactive_instances)))
                    active_instances = Instances.cat(
                        [active_instances, inactive_instances[fp_idx]])

            return active_instances
        else:
            active_mask = track_instances.obj_idxes >= 0
            return track_instances[active_mask]

    def _update_track_embedding(self, track_instances: Instances
                                 ) -> Instances:
        """Update track embeddings via self-attention + FFN."""
        if len(track_instances) == 0:
            return track_instances

        dim = track_instances.query.shape[1]
        query_pos = track_instances.query[:, :dim // 2]
        query_feat = track_instances.query[:, dim // 2:]
        out_embed = track_instances.output_embedding

        # Self-attention
        q = k = query_pos + out_embed
        tgt = out_embed
        tgt2 = self.self_attn(q[None], k[None], value=tgt[None])[0][0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout2(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)

        # Position update
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(F.relu(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)

        # Feature update
        query_feat2 = self.linear_feat2(
            self.dropout_feat1(F.relu(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)

        # Avoid in-place slice assignment for autograd safety
        track_instances.query = torch.cat([query_pos, query_feat], dim=1)

        track_instances.output_embedding = tgt
        return track_instances

    def forward(self, data: dict) -> Instances:
        """Forward pass: select active → update → merge with init.

        Args:
            data: dict with keys:
                - 'track_instances': current frame track instances
                - 'init_track_instances': freshly initialized instances

        Returns:
            Merged Instances with updated active + new init queries.
        """
        track_instances = data['track_instances']
        init_track_instances = data['init_track_instances']

        active_instances = self._select_active_tracks(track_instances)
        active_instances = self._update_track_embedding(active_instances)

        merged = Instances.cat([init_track_instances, active_instances])
        return merged
