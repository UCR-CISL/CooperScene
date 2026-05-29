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
        --data-root /workspace/data/Cooperscene/release/250928_opv2v
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

# Re-use the OPV2V vehicle parser from the existing converter
import sys
_HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _HERE)
from opv2v_converter import parse_opv2v_vehicle  # noqa: E402

EGO_CANDIDATE_IDS = ('1', '2', '3')


def _list_subdirs(p):
    return [d for d in os.listdir(p) if osp.isdir(osp.join(p, d))]


def build_split(data_root: str, split: str, out_path: str) -> int:
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

            pts_filename = f'{timestamp}.pcd'
            pts_abs = osp.join(
                split_path, scenario, agent_id, pts_filename)
            if not osp.exists(pts_abs):
                bin_filename = f'{timestamp}.bin'
                bin_abs = osp.join(
                    split_path, scenario, agent_id, bin_filename)
                if not osp.exists(bin_abs):
                    continue
                pts_filename = bin_filename

            agent_data.append({
                'agent_id': agent_id,
                'ego_pose': pose,
                'lidar_path': osp.join(
                    split, scenario, agent_id, pts_filename),
                'meta': meta,
                'images': {},
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

            # GT bboxes in ego LiDAR frame (parse_opv2v_vehicle requires
            # ego_pose as 6-element [x, y, z, roll, yaw, pitch]).
            instances = []
            vehicles = ego['meta'].get('vehicles', {})
            for veh_id, veh_data in vehicles.items():
                bbox_ego = parse_opv2v_vehicle(veh_data, ego['ego_pose'])
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
                    help='e.g. /workspace/data/Cooperscene/release/250928_opv2v')
    return ap.parse_args()


def main():
    args = parse_args()
    dr = args.data_root
    build_split(dr, 'train',
                osp.join(dr, 'cooperscene_coop_infos_train.pkl'))
    build_split(dr, 'validate',
                osp.join(dr, 'cooperscene_coop_infos_val.pkl'))
    build_split(dr, 'test',
                osp.join(dr, 'cooperscene_coop_infos_test.pkl'))


if __name__ == '__main__':
    main()
