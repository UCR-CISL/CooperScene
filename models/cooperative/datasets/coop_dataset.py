"""CooperScene Cooperative Dataset for 3D Object Detection.

Extends the single-agent CooperSceneDataset to support cooperative perception
with multiple Connected Autonomous Vehicles (CAVs). Each sample contains
ego agent data plus cooperating agents' LiDAR data and poses.
"""

import copy
import itertools
import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .cooperscene_dataset import CooperSceneDataset

# Table 2 networks: ideal/Unlimited = 27 Mbps nominal C-V2X capacity, async OFF
# (fits in one frame, no staleness); C-V2X = 1.6 Mbps measured, async ON.
IDEAL_THROUGHPUT_MBPS = 27.0   # documents the async-off assumption; not numeric
CV2X_THROUGHPUT_MBPS = 1.6     # measured C-V2X throughput for the C-V2X column

# Table 2 agent settings -> (num_peer_vehicles, include_infra). Infra = agent '0'.
AGENT_SETTINGS = {
    'V+I':    (0, True),   # ego vehicle + infra
    'V+V':    (1, False),  # ego vehicle + 1 peer vehicle
    'V+V+I':  (1, True),   # ego vehicle + 1 peer vehicle + infra
    'V+2V':   (2, False),  # ego vehicle + 2 peer vehicles
    'V+2V+I': (2, True),   # ego vehicle + 2 peer vehicles + infra (= all)
}


def pose_to_matrix(pose):
    """Convert pose to a 4x4 transformation matrix.

    Accepts either:
      - CooperScene style: flat list [x, y, z, roll_deg, yaw_deg, pitch_deg]
      - CooperScene style: 4x4 matrix already (list of 4 lists or ndarray)

    Returns:
        4x4 numpy transformation matrix (world frame).
    """
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.ndim == 2 and pose_arr.shape == (4, 4):
        return pose_arr

    x, y, z = pose_arr[0], pose_arr[1], pose_arr[2]
    roll = np.radians(pose_arr[3])
    yaw = np.radians(pose_arr[4])
    pitch = np.radians(pose_arr[5])
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)

    # Must match OpenCOOD's `transformation_utils.x_to_world` (CARLA /
    # CooperScene left-handed convention). A naive right-handed Rz @ Ry @ Rx
    # flips the sign of the pitch/roll terms in the z-row and tilts the
    # cooperator point clouds relative to the ego frame.
    R = np.array([
        [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
        [s_p,       -c_p * s_r,                   c_p * c_r],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


@DATASETS.register_module()
class CoopDataset(CooperSceneDataset):
    """CooperScene Cooperative Dataset.

    Extends single-agent CooperSceneDataset with multi-agent support.
    Loads cooperating agents' LiDAR data and computes transformation
    matrices from each agent's frame to the ego frame.

    Args:
        max_cav: Maximum number of CAVs including ego (default: 5).
        com_range: Communication range in meters (default: 70.0).
    """

    def __init__(self,
                 max_cav: int = 5,
                 com_range: float = 70.0,
                 wild_setting: Optional[dict] = None,
                 agent_setting: Optional[str] = None,
                 network: str = 'unlimited',
                 share_size_mb: float = 0.0,
                 cv2x_throughput: float = CV2X_THROUGHPUT_MBPS,
                 cv2x_async_overhead: float = 0.0,
                 cv2x_backbone_delay: float = 0.0,
                 **kwargs) -> None:
        self.max_cav = max_cav
        self.com_range = com_range

        if agent_setting is not None and agent_setting not in AGENT_SETTINGS:
            raise ValueError(
                f'agent_setting must be one of {list(AGENT_SETTINGS)} or None, '
                f'got {agent_setting!r}')
        self.agent_setting = agent_setting

        # 'unlimited' = perfect sharing (mAP Unlimited); 'cv2x' = async delay
        # from per-cooperator share size / C-V2X throughput (mAP C-V2X).
        self.network = network
        if wild_setting is None and network == 'cv2x':
            wild_setting = {
                'async': True,
                'async_mode': 'real',
                'async_overhead': cv2x_async_overhead,     # ms
                'transmission_speed': cv2x_throughput,      # Mbps
                'data_size': float(share_size_mb) * 8.0,    # MB -> Mb
                'backbone_delay': cv2x_backbone_delay,      # ms
                'loc_err': False,
            }
        self._init_wild_setting(wild_setting)
        super().__init__(**kwargs)

    def _init_wild_setting(self, ws: Optional[dict]) -> None:
        """Parse the OpenCOOD-style `wild_setting` (async transmission delay +
        localization error). Defaults are all-off / backward compatible. Field
        names and semantics mirror OpenCOOD's `basedataset.BaseDataset`."""
        ws = dict(ws or {})
        self.async_flag = bool(ws.get('async', False))
        self.async_mode = ws.get('async_mode', 'sim')
        self.async_overhead = ws.get('async_overhead', 0)        # ms
        self.transmission_speed = ws.get(
            'transmission_speed', IDEAL_THROUGHPUT_MBPS)  # Mbps
        self.data_size = ws.get('data_size', 0)                  # Mb
        self.backbone_delay = ws.get('backbone_delay', 0)        # ms
        self.loc_err_flag = bool(ws.get('loc_err', False))
        self.xyz_noise_std = ws.get('xyz_std', 0.0)              # m
        self.ryp_noise_std = ws.get('ryp_std', 0.0)              # deg
        self.seed = ws.get('seed', 20)
        self._wild_on = self.async_flag or self.loc_err_flag
        self._frame_index = {}       # (scenario, agent_id) -> {ts: entry}
        self._frame_ts_sorted = {}   # (scenario, agent_id) -> sorted [ts]

    def _time_delay(self, is_ego: bool) -> int:
        """Port of OpenCOOD `time_delay_calculation`. Returns the delay in
        frames (data is 10 Hz, so `ms // 100`)."""
        if is_ego:
            return 0
        if self.async_mode == 'real':
            # systematic async noise + data transmission time + backbone compute
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        else:  # 'sim': constant delay
            time_delay = np.abs(self.async_overhead)
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def _add_loc_noise(self, pose):
        """Port of OpenCOOD `add_loc_noise`: gaussian noise on x/y/z and yaw
        (pose[4]); roll/pitch untouched. Seeded with `self.seed`."""
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, self.xyz_noise_std, 3)
        ryp_noise = np.random.normal(0, self.ryp_noise_std, 3)
        return [pose[0] + xyz_noise[0],
                pose[1] + xyz_noise[1],
                pose[2] + xyz_noise[2],
                pose[3],
                pose[4] + ryp_noise[1],
                pose[5]]

    def _build_frame_index(self) -> None:
        """Build a per-(scenario, agent) timestamp -> {lidar_path, pose, images}
        index from the raw annotation file, so async lookups can fetch a
        cooperator's *past* frame. Every agent appears in every frame (as ego
        or cooperator), so the index is complete."""
        from mmengine.fileio import load
        raw = load(self.ann_file)
        raw_list = raw['data_list'] if isinstance(raw, dict) else raw

        def _add(scenario, agent_id, ts, lidar_path, pose, images):
            key = (str(scenario), str(agent_id))
            self._frame_index.setdefault(key, {})[ts] = {
                'lidar_path': lidar_path, 'pose': pose, 'images': images}

        for s in raw_list:
            scenario = s.get('scenario')
            ts = int(round(float(s['timestamp'])))
            _add(scenario, s.get('agent_id'), ts,
                 s['lidar_points']['lidar_path'], s.get('ego_pose'),
                 s.get('images'))
            for c in s.get('cooperators', []):
                _add(scenario, c.get('agent_id'), ts,
                     c['lidar_points']['lidar_path'], c.get('ego_pose'),
                     c.get('images'))
        for key, d in self._frame_index.items():
            self._frame_ts_sorted[key] = sorted(d.keys())

    def _lookup_delayed(self, scenario, agent_id, ts, delay):
        """Return the (scenario, agent) frame `delay` steps before `ts`,
        clamped to the first frame. None if not indexed."""
        key = (str(scenario), str(agent_id))
        ts_list = self._frame_ts_sorted.get(key)
        if not ts_list:
            return None
        ts_i = int(round(float(ts)))
        try:
            pos = ts_list.index(ts_i)
        except ValueError:
            # nearest frame at or before ts
            import bisect
            pos = max(0, bisect.bisect_right(ts_list, ts_i) - 1)
        target = max(0, pos - int(delay))
        return self._frame_index[key][ts_list[target]]

    def load_data_list(self):
        # Build the temporal index BEFORE per-sample parsing so parse_data_info
        # can resolve delayed cooperator frames.
        if self._wild_on:
            self._build_frame_index()
        return super().load_data_list()

    def parse_data_info(self, info: dict):
        """Parse one ego sample. With ``agent_setting`` None, uses all in-range
        cooperators (V+2V+I). Otherwise expands into one sample per cooperator
        combination of that setting (list return; mmengine flattens it)."""
        base = super().parse_data_info(info)
        valid_coops = self._build_valid_coops(info)

        if self.agent_setting is None:
            self._attach_coops(base, valid_coops)
            return base

        out = []
        for subset in self._enumerate_combos(valid_coops):
            di = copy.deepcopy(base)
            self._attach_coops(di, subset)
            out.append(di)
        return out

    def _enumerate_combos(self, valid_coops: List[dict]) -> List[List[dict]]:
        """All cooperator subsets matching ``agent_setting`` (every combo of
        num_peers vehicles, +infra if required); unsatisfiable combos dropped."""
        num_peers, include_infra = AGENT_SETTINGS[self.agent_setting]
        infra = [c for c in valid_coops if str(c['agent_id']) == '0']
        peers = [c for c in valid_coops if str(c['agent_id']) != '0']
        combos = []
        for peer_combo in itertools.combinations(peers, num_peers):
            sel = list(peer_combo)
            if include_infra:
                if not infra:
                    continue
                sel = sel + infra  # infra kept last (matches OpenCOOD ordering)
            combos.append(sel)
        return combos

    def _build_valid_coops(self, info: dict) -> List[dict]:
        """In-range cooperators for an ego sample, with wild-setting (async
        delay + loc error) applied; each entry tagged with its agent_id."""
        ego_pose = info.get('ego_pose', None)
        cooperators = info.get('cooperators', [])

        # Compute transformation matrices and filter by range
        valid_coops = []
        if ego_pose is not None:
            scenario = info.get('scenario')
            cur_ts = info.get('timestamp')
            T_ego = pose_to_matrix(ego_pose)
            ego_pos = T_ego[:3, 3]
            T_ego_inv = np.linalg.inv(T_ego)

            for coop in cooperators:
                coop_pose = coop['ego_pose']

                # Filter by communication range using the CURRENT cooperator
                # position (matches OpenCOOD's calc_dist_to_ego, which always
                # uses the current timestamp, not the delayed one).
                coop_pos = pose_to_matrix(coop_pose)[:3, 3]
                dist = np.linalg.norm(ego_pos[:2] - coop_pos[:2])
                if dist > self.com_range:
                    continue

                # --- wild setting: async transmission delay + loc error ---
                # Substitute the cooperator's PAST frame (lidar + pose) and add
                # localization noise, then transform delayed-cav -> current-ego,
                # mirroring OpenCOOD reform_param (x1_to_x2(delay_cav, cur_ego)).
                src_pose = coop_pose
                src_lidar_rel = coop['lidar_points']['lidar_path']
                src_num_feats = coop['lidar_points'].get('num_pts_feats', 4)
                src_images = coop.get('images')
                if self.async_flag:
                    delay = self._time_delay(is_ego=False)
                    entry = self._lookup_delayed(
                        scenario, coop.get('agent_id'), cur_ts, delay)
                    if entry is not None:
                        src_pose = entry['pose']
                        src_lidar_rel = entry['lidar_path']
                        src_images = entry['images']
                if self.loc_err_flag:
                    src_pose = self._add_loc_noise(src_pose)

                T_cav = pose_to_matrix(src_pose)
                T_cav_to_ego = T_ego_inv @ T_cav

                # Make cooperator lidar_path absolute, matching
                # Det3DDataset.parse_data_info behavior for ego path
                coop_lidar_path = osp.join(
                    self.data_prefix.get('pts', ''), src_lidar_rel)

                coop_entry = {
                    'agent_id': coop.get('agent_id', ''),
                    'lidar_path': coop_lidar_path,
                    'num_pts_feats': src_num_feats,
                    'transformation_matrix': T_cav_to_ego.astype(
                        np.float32),
                    'dist': float(dist),
                }

                # Thread cooperator camera images if available
                if src_images:
                    # Make image paths absolute (like lidar_path above)
                    coop_images = {}
                    for cam_name, cam_info in src_images.items():
                        abs_cam_info = dict(cam_info)
                        abs_cam_info['img_path'] = osp.join(
                            self.data_prefix.get('pts', ''),
                            cam_info['img_path'])
                        coop_images[cam_name] = abs_cam_info
                    coop_entry['images'] = coop_images

                valid_coops.append(coop_entry)

            # OpenCOOD cav ordering: vehicles by id ascending, infra ('0') last.
            valid_coops.sort(
                key=lambda x: (1 if str(x['agent_id']) == '0' else 0,
                               int(x['agent_id'])))

        return valid_coops

    def _attach_coops(self, data_info: dict, valid_coops: List[dict]) -> dict:
        """Fill cooperator infos / (max_cav,4,4) transforms / mask from the
        selected cooperator list (slot 0 = ego identity, unused slots = eye)."""
        t_matrix = np.tile(
            np.eye(4, dtype=np.float32), (self.max_cav, 1, 1))
        coop_mask = np.zeros(self.max_cav, dtype=bool)
        coop_mask[0] = True

        coop_infos = []
        for i, coop in enumerate(valid_coops[:self.max_cav - 1]):
            idx = i + 1
            t_matrix[idx] = coop['transformation_matrix']
            coop_mask[idx] = True
            coop_entry = {
                'agent_id': coop.get('agent_id', ''),
                'lidar_path': coop['lidar_path'],
                'num_pts_feats': coop['num_pts_feats'],
            }
            if 'images' in coop and coop['images']:
                coop_entry['images'] = coop['images']
            coop_infos.append(coop_entry)

        data_info['cooperators'] = coop_infos
        data_info['transformation_matrix'] = t_matrix
        data_info['coop_mask'] = coop_mask
        return data_info
