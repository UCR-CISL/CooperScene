"""OPV2V Motion Dataset for trajectory prediction.

Extends OPV2VCoopDataset with:
- Future/past trajectory GT per vehicle
- Map labels (vectorized polylines from LiDARInstanceLines)
- Temporal sequence metadata

Requires info pkl pre-processed by opv2v_trajectory_converter.py.
"""

import os.path as osp
import pickle
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .opv2v_coop_dataset import OPV2VCoopDataset


@DATASETS.register_module()
class OPV2VMotionDataset(OPV2VCoopDataset):
    """OPV2V dataset with motion prediction GT.

    Loads pre-computed trajectory GT (future/past) and optional map labels.

    Args:
        future_steps (int): Number of future trajectory steps. Default: 12.
        past_steps (int): Number of past trajectory steps. Default: 4.
        map_labels_path (str): Path to map_labels.pkl. Default: None.
        max_agents (int): Maximum agents per frame for padding. Default: 50.
        queue_length (int): Temporal queue length for multi-frame training.
            1 = single-frame (default). >1 = returns queue of consecutive
            frames using prev_idx pointers.
    """

    def __init__(self,
                 future_steps: int = 12,
                 past_steps: int = 4,
                 map_labels_path: Optional[str] = None,
                 max_agents: int = 50,
                 queue_length: int = 1,
                 **kwargs) -> None:
        self.future_steps = future_steps
        self.past_steps = past_steps
        self.max_agents = max_agents
        self.map_labels_path = map_labels_path
        self.queue_length = queue_length
        self._map_labels = None  # lazy loaded
        self._prev_idx_map = {}  # built after super().__init__

        super().__init__(**kwargs)

    def full_init(self):
        """Override to build temporal index before data serialization.

        mmengine's BaseDataset.full_init() serializes data_list into bytes
        and clears the list. We must build the temporal index while
        data_list is still accessible as a plain list of dicts.
        """
        if self._fully_initialized:
            return

        # Run parent full_init steps: load_data_list + filter_data + indices
        self.data_list = self.load_data_list()
        self.data_list = self.filter_data()
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # Build temporal index NOW while data_list is still a plain list
        if self.queue_length > 1:
            self._build_temporal_index()

        # Serialize data (clears self.data_list)
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    @property
    def map_labels(self):
        """Lazy-load map labels to avoid loading at import time."""
        if self._map_labels is None and self.map_labels_path is not None:
            print(f'Loading map labels from {self.map_labels_path}...')
            # Need to handle LiDARInstanceLines pickle
            import sys
            # Add a mock if the actual class isn't importable
            try:
                from projects.GenAD.genad.map_utils import LiDARInstanceLines
            except ImportError:
                # Create a simple mock that preserves data
                class LiDARInstanceLines:
                    def __init__(self, *args, **kwargs):
                        pass
                # Register in pickle's namespace
                import types
                mock_module = types.ModuleType('mmdet3d.datasets.map_utils')
                mock_module.LiDARInstanceLines = LiDARInstanceLines
                sys.modules['mmdet3d.datasets.map_utils'] = mock_module

            with open(self.map_labels_path, 'rb') as f:
                self._map_labels = pickle.load(f)
            print(f'Loaded map labels: {len(self._map_labels)} scenarios')
        return self._map_labels

    def parse_data_info(self, info: dict) -> dict:
        """Parse info dict to add trajectory and map data.

        Extends parent's parse_data_info with:
        - gt_fut_traj: (N, future_steps, 2) future trajectories
        - gt_fut_traj_mask: (N, future_steps) validity mask
        - gt_past_traj: (N, past_steps, 2) past trajectories
        - gt_past_traj_mask: (N, past_steps) validity mask
        - gt_vehicle_ids: list of vehicle IDs

        Important: parent's parse_ann_info filters gt_bboxes_3d by
        pcd_limit_range. We must apply the same mask to trajectory
        data so indices stay aligned.
        """
        # Recompute the pcd_limit_range mask BEFORE super() filters bboxes,
        # so we can apply it to trajectory data as well.
        instances = info.get('instances', [])
        if len(instances) > 0:
            bboxes_raw = np.array([inst['bbox_3d'] for inst in instances])
            pcd_range = np.array(self.pcd_limit_range)
            range_mask = (
                (bboxes_raw[:, 0] >= pcd_range[0]) &
                (bboxes_raw[:, 0] <= pcd_range[3]) &
                (bboxes_raw[:, 1] >= pcd_range[1]) &
                (bboxes_raw[:, 1] <= pcd_range[4]) &
                (bboxes_raw[:, 2] >= pcd_range[2]) &
                (bboxes_raw[:, 2] <= pcd_range[5])
            )
        else:
            range_mask = np.ones(0, dtype=bool)

        data_info = super().parse_data_info(info)

        # Trajectory GT — apply same range mask as gt_bboxes_3d
        if 'gt_fut_traj' in info:
            data_info['gt_fut_traj'] = info['gt_fut_traj'][range_mask].astype(
                np.float32)
            data_info['gt_fut_traj_mask'] = info[
                'gt_fut_traj_mask'][range_mask].astype(np.float32)
        else:
            data_info['gt_fut_traj'] = np.zeros(
                (0, self.future_steps, 2), dtype=np.float32)
            data_info['gt_fut_traj_mask'] = np.zeros(
                (0, self.future_steps), dtype=np.float32)

        if 'gt_past_traj' in info:
            data_info['gt_past_traj'] = info[
                'gt_past_traj'][range_mask].astype(np.float32)
            data_info['gt_past_traj_mask'] = info[
                'gt_past_traj_mask'][range_mask].astype(np.float32)
        else:
            data_info['gt_past_traj'] = np.zeros(
                (0, self.past_steps, 2), dtype=np.float32)
            data_info['gt_past_traj_mask'] = np.zeros(
                (0, self.past_steps), dtype=np.float32)

        gt_vids = info.get('gt_vehicle_ids', [])
        if len(gt_vids) > 0:
            data_info['gt_vehicle_ids'] = np.array(
                gt_vids)[range_mask].tolist()
        else:
            data_info['gt_vehicle_ids'] = []

        # Temporal pointers
        data_info['prev_idx'] = info.get('prev_idx', -1)
        data_info['next_idx'] = info.get('next_idx', -1)

        # BEV map PNG paths (constructed from scenario/agent_id/timestamp)
        lidar_path = info.get('lidar_points', {}).get('lidar_path', '')
        if lidar_path:
            split = lidar_path.split('/')[0]
            scenario = str(info.get('scenario', ''))
            agent_id = str(info.get('agent_id', ''))
            timestamp = int(info.get('timestamp', 0))
            ts_str = f'{timestamp:06d}'
            base_dir = osp.join(self.data_root, split, scenario, agent_id)
            data_info['bev_map_paths'] = {
                'bev_lane': osp.join(base_dir, f'{ts_str}_bev_lane.png'),
                'bev_static': osp.join(base_dir, f'{ts_str}_bev_static.png'),
                'bev_dynamic': osp.join(base_dir, f'{ts_str}_bev_dynamic.png'),
            }
        else:
            data_info['bev_map_paths'] = {}

        # Map labels (optional)
        if self.map_labels is not None:
            scenario = str(info.get('scenario', ''))
            agent_id = str(info.get('agent_id', ''))
            timestamp = str(int(info.get('timestamp', 0)))

            map_data = None
            if scenario in self.map_labels:
                agent_map = self.map_labels[scenario]
                if agent_id in agent_map:
                    ts_map = agent_map[agent_id]
                    if timestamp in ts_map:
                        map_data = ts_map[timestamp]

            if map_data is not None:
                data_info['map_gt'] = map_data
            else:
                data_info['map_gt'] = None

        return data_info

    def _build_temporal_index(self):
        """Build temporal prev-frame mapping on the FILTERED data_list.

        After filter_data() removes empty-GT frames, the original prev_idx
        from the pkl is invalid. Rebuild using (scenario, agent_id, timestamp).
        """
        from collections import defaultdict
        scene_frames = defaultdict(list)
        for filtered_idx, info in enumerate(self.data_list):
            key = (str(info.get('scenario', '')),
                   str(info.get('agent_id', '')))
            ts = int(info.get('timestamp', -1))
            scene_frames[key].append((ts, filtered_idx))

        for key in scene_frames:
            scene_frames[key].sort()

        self._prev_idx_map = {}
        for key, frames in scene_frames.items():
            for i, (ts, fidx) in enumerate(frames):
                if i > 0:
                    self._prev_idx_map[fidx] = frames[i - 1][1]

        print(f'Built temporal index: {len(self._prev_idx_map)} frames '
              f'have previous frames, {len(scene_frames)} sequences')

    def _get_queue_indices(self, idx: int) -> list:
        """Get temporal queue frame indices: [t-(Q-1), ..., t-1, t].

        Uses _prev_idx_map built on the filtered data_list.
        Pads with earliest available frame if history is too short.
        """
        indices = [idx]
        curr_idx = idx
        for _ in range(self.queue_length - 1):
            prev_idx = self._prev_idx_map.get(curr_idx, None)
            if prev_idx is not None:
                indices.insert(0, prev_idx)
                curr_idx = prev_idx
            else:
                indices.insert(0, indices[0])
        return indices

    def prepare_data(self, idx: int):
        """Prepare data, optionally packing a temporal queue.

        For queue_length > 1 (training only):
        - Processes Q consecutive frames through the pipeline
        - Packs them into a TemporalQueue wrapper
        - Returns the last (current) frame as the primary result

        The TemporalQueue is an opaque wrapper that pseudo_collate
        won't recurse into, preserving the multi-frame structure
        for the QueueDet3DDataPreprocessor to handle.
        """
        if self.queue_length <= 1 or self.test_mode:
            return super().prepare_data(idx)

        from projects.UniAD.uniad.data_preprocessor import TemporalQueue

        indices = self._get_queue_indices(idx)

        queue_data = []
        for fidx in indices:
            data = super().prepare_data(fidx)
            if data is None:
                return None  # trigger retry in __getitem__
            queue_data.append(data)

        # Last frame is the current frame — use as primary result
        result = queue_data[-1]
        result['temporal_queue'] = TemporalQueue(queue_data)
        return result
