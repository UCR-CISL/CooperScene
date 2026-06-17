"""Build CooperScene train+val coop pkls with infra agent 0 as cooperator
but only vehicles 1/2/3 ever taking the ego role.

Reads from <data_root>/{train,validate}/<scenario>/<agent>/<timestamp>.{pcd,yaml}
Writes:
    <data_root>/cooperscene_coop_infos_train.pkl
    <data_root>/cooperscene_coop_infos_val.pkl

Each (scenario, timestamp) produces 3 samples — one per vehicle ego, with
the other 2 vehicles + agent 0 (infra) as cooperators.

Usage:
    python tools/dataset_converters/cooperscene_train_val_converter.py \
        --data-root /workspace/data/Cooperscene/release/250928_cooperscene
"""
import argparse
import os
from collections import defaultdict
from os import path as osp
from typing import List

import mmengine
import numpy as np
import yaml
from tqdm import tqdm

# Re-use the CooperScene vehicle parser from the existing converter
import sys
_HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _HERE)
from data_converter import parse_cooperscene_vehicle, convert_pcd_to_bin  # noqa: E402

EGO_CANDIDATE_IDS = ('1', '2', '3')

# Communication range (m): cavs farther than this from the ego don't
# cooperate, matching OpenCOOD's COM_RANGE and CoopDataset.com_range.
COM_RANGE = 70.0


# Per-vehicle camera calibration (front camera, "camera0").
# Source: SensorPlatform/utils/constants.py (CALIBRATION).
# `intrinsic` is the 3x3 pinhole matrix; `T_lidar_to_cam` is the 4x4
# lidar->camera transform already in CV convention (X right, Y down,
# Z forward), so no CARLA->CV swap is needed.
AGENT_CALIBRATION = {
    '1': {
        'intrinsic': [
            [1810.6983377214349, 0.0, 970.0037900558976],
            [0.0, 1808.4746114412712, 526.7243033929532],
            [0.0, 0.0, 1.0],
        ],
        'lidar2cam': [
            [ 1.97675856e-02, -9.98998871e-01, -4.01310140e-02, -1.14236290e-04],
            [-4.47204289e-02,  3.92152089e-02, -9.98229558e-01, -1.75248879e-01],
            [ 9.98803948e-01,  2.15272644e-02, -4.39004680e-02, -5.25987245e-02],
            [ 0.0, 0.0, 0.0, 1.0],
        ],
    },
    '2': {
        'intrinsic': [
            [1807.294188309111, 0.0, 959.1274184107824],
            [0.0, 1805.1344921703185, 562.0235332672737],
            [0.0, 0.0, 1.0],
        ],
        'lidar2cam': [
            [ 0.00383388, -0.99998521, -0.00385735, -0.0013977],
            [-0.00271154,  0.00384697, -0.99998892, -0.23015363],
            [ 0.99998897,  0.0038443,  -0.00269675, -0.07767752],
            [ 0.0, 0.0, 0.0, 1.0],
        ],
    },
    '3': {
        'intrinsic': [
            [1814.1641900346326, 0.0, 951.718698093622],
            [0.0, 1812.1352460466578, 568.1954332460546],
            [0.0, 0.0, 1.0],
        ],
        'lidar2cam': [
            [ 0.064023,   -0.99792386, -0.00700149, -0.00332843],
            [-0.01444825,  0.00608825, -0.99987708, -0.26612881],
            [ 0.99784383,  0.06411629, -0.01402846, -0.0864176],
            [ 0.0, 0.0, 0.0, 1.0],
        ],
    },
}


def _extract_images_for_ts(meta, split, scenario, agent_id, timestamp,
                           data_root):
    """Per-agent `images` dict (img_path, lidar2cam, cam2img). CooperScene
    release ships front camera frames as `<ts>_camera0.png` for vehicle
    agents {1, 2, 3}; intrinsic / extrinsic are not in the yaml so we
    look them up from `AGENT_CALIBRATION`. Agent 0 (LiDAR-only RSU) has
    no entry and returns an empty dict."""
    images = {}
    calib = AGENT_CALIBRATION.get(str(agent_id))
    if calib is None:
        return images
    cam_img_filename = f'{timestamp}_camera0.png'
    cam_img_path = osp.join(
        data_root, split, scenario, agent_id, cam_img_filename)
    if not osp.exists(cam_img_path):
        return images
    images['camera0'] = {
        'img_path': osp.join(
            split, scenario, agent_id, cam_img_filename),
        'lidar2cam': calib['lidar2cam'],
        'cam2img': calib['intrinsic'],
    }
    return images


def _list_subdirs(p):
    return [d for d in os.listdir(p) if osp.isdir(osp.join(p, d))]


def build_split(data_root: str, split: str, out_path: str,
                convert_pcd: bool = False,
                bin_dir: str = None) -> int:
    split_path = osp.join(data_root, split)
    if not osp.exists(split_path):
        print(f'  SKIP {split}, not found')
        return 0

    # group: {(scenario, timestamp): [(yaml_path, agent_id)]}
    groups = defaultdict(list)
    scenarios = sorted(_list_subdirs(split_path))
    print(f'[{split}] {len(scenarios)} scenarios: {scenarios}')

    for scenario in scenarios:
        scen_p = osp.join(split_path, scenario)
        for agent_id in sorted(_list_subdirs(scen_p)):
            agent_p = osp.join(scen_p, agent_id)
            for f in os.listdir(agent_p):
                if (f.endswith('.yaml')
                        and not f.startswith('data_protocol')
                        and not f.endswith('_additional.yaml')
                        and not f.endswith('_separate.yaml')):
                    timestamp = f[:-5]
                    groups[(scenario, timestamp)].append(
                        (osp.join(agent_p, f), agent_id))

    print(f'  {len(groups)} (scenario, timestamp) groups')

    infos = []
    sample_idx = 0

    for (scenario, timestamp), entries in tqdm(
            sorted(groups.items()), desc=f'{split}'):
        # Parse every agent at this (scenario, timestamp)
        agent_data = []
        for yaml_path, agent_id in entries:
            try:
                with open(yaml_path, 'r') as f:
                    meta = yaml.unsafe_load(f)
            except Exception:
                continue
            pose = meta.get('lidar_pose', meta.get('true_ego_pos'))
            if pose is None:
                continue

            # Prefer .bin if it exists, fall back to .pcd. mmdet3d's
            # `LoadPointsFromFile` only handles raw float32 (.bin); the
            # cooperative loader can read .pcd directly, but bevfusion
            # uses LoadPointsFromFile, so for that pipeline we want
            # `.bin`. With --convert-pcd, build a .bin (next to the .pcd
            # by default, or under `--bin-dir` mirroring the original
            # tree); pkl lidar_path stays relative when .bin is next to
            # .pcd, and becomes absolute when --bin-dir is in use.
            pcd_filename = f'{timestamp}.pcd'
            pcd_abs = osp.join(split_path, scenario, agent_id, pcd_filename)
            inplace_bin = osp.join(
                split_path, scenario, agent_id, f'{timestamp}.bin')
            if bin_dir is not None:
                external_bin = osp.join(
                    bin_dir, split, scenario, agent_id, f'{timestamp}.bin')
                if convert_pcd and osp.exists(pcd_abs) \
                        and not osp.exists(external_bin):
                    os.makedirs(osp.dirname(external_bin), exist_ok=True)
                    convert_pcd_to_bin(pcd_abs, external_bin)
                if osp.exists(external_bin):
                    pts_filename = external_bin  # absolute path
                elif osp.exists(inplace_bin):
                    pts_filename = f'{timestamp}.bin'  # relative to split
                elif osp.exists(pcd_abs):
                    pts_filename = pcd_filename
                else:
                    continue
            else:
                if convert_pcd and osp.exists(pcd_abs) \
                        and not osp.exists(inplace_bin):
                    convert_pcd_to_bin(pcd_abs, inplace_bin)
                if osp.exists(inplace_bin):
                    pts_filename = f'{timestamp}.bin'
                elif osp.exists(pcd_abs):
                    pts_filename = pcd_filename
                else:
                    continue

            images = _extract_images_for_ts(
                meta, split, scenario, agent_id, timestamp, data_root)
            agent_data.append({
                'agent_id': agent_id,
                'ego_pose': pose,
                'lidar_path': osp.join(
                    split, scenario, agent_id, pts_filename),
                'meta': meta,
                'images': images,
            })

        if not agent_data:
            continue

        agent_data.sort(key=lambda x: x['agent_id'])

        # Only rotate vehicles (id in {'1','2','3'}) as ego
        for ego in agent_data:
            if ego['agent_id'] not in EGO_CANDIDATE_IDS:
                continue

            cooperators = [a for a in agent_data
                           if a['agent_id'] != ego['agent_id']]

            # GT bboxes in ego LiDAR frame (parse_cooperscene_vehicle requires
            # ego_pose as 6-element [x, y, z, roll, yaw, pitch]).
            #
            # Cooperative GT is the UNION of every agent's visible vehicles
            # (deduplicated by vehicle id), matching OpenCOOD's
            # IntermediateFusionDataset which merges object_bbx_center across
            # all CAVs. Using only the ego's own `vehicles` misses vehicles
            # that the ego can't see but cooperators can - those then count as
            # false positives at eval and depress AP. Iterate ego-first so the
            # ego's pose is kept for vehicles seen by multiple agents.
            # Only cavs within the communication range contribute GT, matching
            # OpenCOOD (its GT collection sits inside the COM_RANGE filter).
            ego_xy = np.asarray(ego['ego_pose'][:2], dtype=np.float64)
            vehicles = {}
            for a in [ego] + cooperators:
                if a is not ego:
                    a_xy = np.asarray(a['ego_pose'][:2], dtype=np.float64)
                    if np.linalg.norm(a_xy - ego_xy) > COM_RANGE:
                        continue
                for vid, vdata in a['meta'].get('vehicles', {}).items():
                    if vid not in vehicles:
                        vehicles[vid] = vdata

            instances = []
            for veh_id, veh_data in vehicles.items():
                bbox_ego = parse_cooperscene_vehicle(veh_data, ego['ego_pose'])
                if bbox_ego is None:
                    continue
                instances.append({
                    'bbox_3d': bbox_ego,
                    'bbox_label_3d': 0,
                })

            infos.append({
                'sample_idx': sample_idx,
                'scenario': scenario,
                'timestamp': float(timestamp),
                'agent_id': ego['agent_id'],
                'ego_pose': ego['ego_pose'],
                'lidar_points': {
                    'lidar_path': ego['lidar_path'],
                    'num_pts_feats': 4,
                },
                'instances': instances,
                'cooperators': [
                    {
                        'agent_id': c['agent_id'],
                        'ego_pose': c['ego_pose'],
                        'lidar_points': {
                            'lidar_path': c['lidar_path'],
                            'num_pts_feats': 4,
                        },
                        'images': c['images'],
                    }
                    for c in cooperators
                ],
                'images': ego['images'],
            })
            sample_idx += 1

    mmengine.dump({
        'data_list': infos,
        'metainfo': {
            'classes': ('vehicle',),
            'categories': {'vehicle': 0},
            'dataset': 'CooperScene',
            'info_version': '1.0',
            'cooperative': True,
        },
    }, out_path)
    print(f'  -> wrote {len(infos)} samples to {out_path}')
    return len(infos)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True,
                    help='e.g. /workspace/data/Cooperscene/release/250928_cooperscene')
    ap.add_argument('--convert-pcd', action='store_true',
                    help='Convert each .pcd to .bin so mmdet3d'
                         ' LoadPointsFromFile (bevfusion path) can read it.')
    ap.add_argument('--bin-dir', default=None,
                    help='Optional output dir for the converted .bin files'
                         ' (mirrors <split>/<scenario>/<agent>/ tree). When'
                         ' unset, .bin lives next to the .pcd.')
    return ap.parse_args()


def main():
    args = parse_args()
    dr = args.data_root
    cp = args.convert_pcd
    bd = args.bin_dir
    build_split(dr, 'train',
                osp.join(dr, 'cooperscene_coop_infos_train.pkl'), cp, bd)
    build_split(dr, 'validate',
                osp.join(dr, 'cooperscene_coop_infos_val.pkl'), cp, bd)
    build_split(dr, 'test',
                osp.join(dr, 'cooperscene_coop_infos_test.pkl'), cp, bd)


if __name__ == '__main__':
    main()
