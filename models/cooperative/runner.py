"""Cooperative-perception training / evaluation runner.

For cooperative detectors (the 4 intermediate-fusion models) we bypass
mmengine's Runner and instead use the original cooperative-fusion training
loop, dataset and post-processor directly. This guarantees that given the
same data + checkpoint, the numbers produced here are bit-for-bit identical
to what the upstream cooperative-perception repo produces.

The mmdet3d-style config is translated in-memory to the dict that the
upstream pipeline expects, and the entry-points below (`train`, `test`)
are called by `tools/train.py` / `tools/test.py` after they detect a
cooperative-detector config.

A small `RotatingEgoDataset` subclass lets us train / evaluate over
multiple ego choices per (scenario, timestamp) instead of the upstream
default of `min(cav_id)`.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import statistics
import time
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


_ARCH_TO_CORE = {
    'v2vam':  'point_pillar_intermediate_V2VAM',
    'cobevt': 'point_pillar_cobevt',
    'v2vnet': 'point_pillar_v2vnet',
    'v2xvit': 'point_pillar_transformer',
}


# ---------------------------------------------------------------------------
# Dataset subclass: rotate ego across {1, 2, 3} per (scenario, timestamp).
# ---------------------------------------------------------------------------
def _make_rotating_ego_dataset_class():
    """Lazy import so test/train scripts don't pull opencood when running
    bevfusion configs."""
    from opencood.data_utils.datasets.intermediate_fusion_dataset import (
        IntermediateFusionDataset)

    class RotatingEgoDataset(IntermediateFusionDataset):
        """Each (scenario, timestamp) yields one sample per ego candidate."""

        def __init__(self, params, visualize=False, train=True,
                     ego_candidates: Optional[Sequence[str]] = None):
            super().__init__(params, visualize=visualize, train=train)
            self.ego_candidates = (
                [str(c) for c in ego_candidates] if ego_candidates else None)
            self._base_len = super().__len__()

        def __len__(self):
            if self.ego_candidates is None:
                return self._base_len
            return self._base_len * len(self.ego_candidates)

        def __getitem__(self, idx):
            if self.ego_candidates is None:
                return super().__getitem__(idx)
            n = len(self.ego_candidates)
            base_idx = idx // n
            ego_id = self.ego_candidates[idx % n]
            self._mark_ego(base_idx, ego_id)
            return super().__getitem__(base_idx)

        def _mark_ego(self, base_idx: int, ego_id: str) -> None:
            """Set scenario_database[scenario]['<ego_id>']['ego'] = True and
            all other cav_ids in that scenario to False.

            DataLoader workers each get a forked copy of the dataset, so
            mutating scenario_database here is safe across workers (no shared
            state) and within a worker (sequential __getitem__)."""
            scenario_idx = None
            for i, cum_len in enumerate(self.len_record):
                if base_idx < cum_len:
                    scenario_idx = i
                    break
            if scenario_idx is None:
                return
            scen = self.scenario_database[scenario_idx]
            for cav_id in scen:
                scen[cav_id]['ego'] = (str(cav_id) == ego_id)
            # OpenCOOD's IntermediateFusionDataset asserts the ego cav is the
            # FIRST key in base_data_dict (built in scenario_database order).
            # The default ego is the min cav_id (agent 0 / infra); when we
            # rotate ego onto a vehicle (1/2/3) we must move it to the front.
            ego_key = next(
                (k for k in scen if str(k) == ego_id), None)
            if ego_key is not None and next(iter(scen)) != ego_key:
                reordered = OrderedDict([(ego_key, scen[ego_key])])
                for k, v in scen.items():
                    if k != ego_key:
                        reordered[k] = v
                self.scenario_database[scenario_idx] = reordered

    return RotatingEgoDataset


# ---------------------------------------------------------------------------
# Config translation: our mmdet3d cfg -> upstream hypes dict.
# ---------------------------------------------------------------------------
def _build_hypes(cfg, *, for_train: bool) -> dict:
    arch = cfg.model.arch
    if arch not in _ARCH_TO_CORE:
        raise ValueError(
            f'unknown arch {arch!r}; expected one of {list(_ARCH_TO_CORE)}')

    voxel_size = list(cfg.voxel_size)
    pcr = list(cfg.point_cloud_range)
    max_cav = int(cfg.model.max_cav)

    # data_root from cfg; append split sub-dir
    train_root = osp.join(cfg.train_dataloader.dataset.data_root, 'train') \
        if for_train else None
    val_root = osp.join(cfg.val_dataloader.dataset.data_root, 'validate')
    test_root = osp.join(cfg.test_dataloader.dataset.data_root, 'test')

    max_epochs = int(getattr(cfg, 'train_cfg', {}).get('max_epochs', 60)) \
        if hasattr(cfg, 'train_cfg') else 60

    train_bs = int(cfg.train_dataloader.batch_size) if for_train else 1

    # Voxel preprocess caps come from the data_preprocessor (mmdet3d style).
    dp = cfg.model.data_preprocessor
    preprocess_args = {
        'voxel_size': voxel_size,
        'max_points_per_voxel': int(dp.max_points_per_voxel),
        'max_voxel_train': int(dp.max_voxel_train),
        'max_voxel_test': int(dp.max_voxel_test),
    }

    # Top-level fields holding the upstream-flavored model / anchor / loss.
    opencood_args = dict(cfg.opencood_args)
    anchor_args = dict(cfg.opencood_anchor_args)
    post_args = dict(cfg.opencood_postprocess_args)
    loss_args = dict(cfg.opencood_loss_args)

    hypes = {
        'name': arch,
        'yaml_parser': 'load_point_pillar_params',
        'root_dir': train_root or val_root,
        'validate_dir': val_root,
        'test_dir': test_root,

        # Coordinate system flag in the upstream base dataset; CooperScene / Carla
        # data is left-handed (not ENU), so this stays False.
        'useENU': False,
        # Read .pcd intensity (int32 / 65535) instead of constant 1.0.
        # Matches what the upstream models were trained with.
        'intensity': True,

        'wild_setting': {
            'async': False, 'async_overhead': 0, 'seed': 20,
            'loc_err': False, 'xyz_std': 0.0, 'ryp_std': 0.0,
            'data_size': 0, 'transmission_speed': 0, 'backbone_delay': 0,
        },

        'train_params': {
            'batch_size': train_bs,
            'epoches': max_epochs,
            'eval_freq': 1,
            'save_freq': 1,
            'max_cav': max_cav,
        },

        'fusion': {
            'core_method': 'IntermediateFusionDataset',
            'args': [],
        },

        'preprocess': {
            'core_method': 'SpVoxelPreprocessor',
            'args': preprocess_args,
            'cav_lidar_range': pcr,
        },

        'data_augment': [
            {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']},
            {'NAME': 'random_world_rotation',
             'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
            {'NAME': 'random_world_scaling',
             'WORLD_SCALE_RANGE': [0.95, 1.05]},
        ],

        'postprocess': {
            'core_method': 'VoxelPostprocessor',
            'anchor_args': anchor_args,
            'target_args': dict(post_args['target_args']),
            'order': 'hwl',
            'max_num': int(post_args['max_num']),
            'nms_thresh': float(post_args['nms_thresh']),
            'gt_range': pcr,
        },

        'model': {
            'core_method': _ARCH_TO_CORE[arch],
            'args': opencood_args,
        },

        'loss': {
            'core_method': 'point_pillar_loss',
            'args': loss_args,
        },

        'optimizer': {
            'core_method': 'Adam',
            'lr': float(cfg.optim_wrapper.optimizer.lr),
            'args': {
                'eps': float(cfg.optim_wrapper.optimizer.eps),
                'weight_decay': float(cfg.optim_wrapper.optimizer.weight_decay),
            },
        },

        'lr_scheduler': {
            'core_method': 'cosineannealwarm',
            'epoches': max_epochs,
            'warmup_lr': 2e-4,
            'warmup_epoches': 10,
            'lr_min': 2e-5,
        },
    }
    return hypes


# ---------------------------------------------------------------------------
# Model + ckpt helpers.
# ---------------------------------------------------------------------------
def _build_model(hypes):
    """Same logic as upstream `train_utils.create_model`, inlined to avoid a
    side-effecting import path."""
    import importlib
    core_method = hypes['model']['core_method']
    model_lib = importlib.import_module(f'opencood.models.{core_method}')
    target = core_method.replace('_', '').lower()
    cls = None
    for name, obj in model_lib.__dict__.items():
        if name.lower() == target:
            cls = obj
            break
    if cls is None:
        raise RuntimeError(
            f'model class for {core_method!r} not found in opencood.models')
    return cls(hypes['model']['args'])


def _load_ckpt(model, ckpt_path: str, *, strict: bool = False):
    """Load a checkpoint into the upstream model.

    Accepts both raw upstream `state_dict` and our wrapper-format ckpts where
    every key is prefixed with `opencood.` (and the unused `bbox_head.*` keys
    are skipped). Returns the (missing, unexpected) tuple from PyTorch.
    """
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    cleaned = {}
    for k, v in state.items():
        if k.startswith('opencood.'):
            cleaned[k[len('opencood.'):]] = v
        elif k.startswith('bbox_head.'):
            # mmdet3d-side head, not part of the upstream model
            continue
        else:
            cleaned[k] = v
    return model.load_state_dict(cleaned, strict=strict)


# ---------------------------------------------------------------------------
# Dataset construction with optional ego rotation.
# ---------------------------------------------------------------------------
def _build_dataset(hypes, *, train: bool, ego_candidates):
    Rotating = _make_rotating_ego_dataset_class()
    return Rotating(hypes, visualize=False, train=train,
                    ego_candidates=ego_candidates)


# ---------------------------------------------------------------------------
# Public entry: train.
# ---------------------------------------------------------------------------
def train(cfg, args: argparse.Namespace) -> None:
    """Cooperative-perception training loop. Mirrors the upstream train.py
    but reads everything (data paths, hypes, batch size, lr, load_from) from
    our mmdet3d cfg."""
    import tqdm
    from tensorboardX import SummaryWriter

    from opencood.tools import train_utils

    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)

    hypes = _build_hypes(cfg, for_train=True)

    ego_candidates = cfg.get('ego_candidates', None)

    print('-----------------Dataset Building------------------')
    train_set = _build_dataset(hypes, train=True, ego_candidates=ego_candidates)
    val_set = _build_dataset(hypes, train=False, ego_candidates=ego_candidates)
    print(f'  train: {len(train_set)} samples '
          f'(base {train_set._base_len} x ego {ego_candidates or 1})')
    print(f'  val:   {len(val_set)} samples')

    batch_size = hypes['train_params']['batch_size']
    num_workers = int(cfg.train_dataloader.num_workers)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers,
        collate_fn=train_set.collate_batch_train, shuffle=True,
        pin_memory=False, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=num_workers,
        collate_fn=train_set.collate_batch_train, shuffle=False,
        pin_memory=False, drop_last=True)

    print('---------------Creating Model------------------')
    model = _build_model(hypes)

    init_epoch = 0
    if getattr(cfg, 'load_from', None):
        print(f'Loading weights from {cfg.load_from}')
        _load_ckpt(model, cfg.load_from, strict=False)
    if getattr(cfg, 'resume', False):
        # auto-resume from the latest net_epoch*.pth in work_dir
        init_epoch, model = train_utils.load_saved_model(work_dir, model)
        print(f'Resumed at epoch {init_epoch}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)
    writer = SummaryWriter(work_dir)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        else:
            scheduler.step_update(epoch * num_steps + 0)
        for pg in optimizer.param_groups:
            print('learning rate %.7f' % pg['lr'])

        pbar = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)

            output_dict = model(batch_data['ego'])
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar)
            pbar.update(1)

            final_loss.backward()
            optimizer.step()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
        pbar.close()

        if epoch % hypes['train_params']['save_freq'] == 0:
            ckpt_path = osp.join(work_dir, f'net_epoch{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  saved {ckpt_path}')

        if epoch % hypes['train_params']['eval_freq'] == 0:
            val_losses = []
            with torch.no_grad():
                for batch_data in val_loader:
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])
                    final_loss = criterion(
                        output_dict, batch_data['ego']['label_dict'])
                    val_losses.append(final_loss.item())
            mean_loss = statistics.mean(val_losses) if val_losses else float('nan')
            print(f'At epoch {epoch}, the validation loss is {mean_loss:.4f}')
            writer.add_scalar('Validate_Loss', mean_loss, epoch)

    print(f'Training Finished, checkpoints saved to {work_dir}')


# ---------------------------------------------------------------------------
# Public entry: test (inference + AP).
# ---------------------------------------------------------------------------
def test(cfg, args: argparse.Namespace) -> None:
    """Cooperative-perception evaluation loop. Mirrors the upstream
    inference.py for `--fusion_method intermediate`."""
    from tqdm import tqdm

    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils

    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)

    hypes = _build_hypes(cfg, for_train=False)

    ego_candidates = cfg.get('ego_candidates', None)

    # For testing, the dataset's `train=False` path reads `validate_dir`. Swap
    # in the test root so we evaluate on the test split.
    hypes['validate_dir'] = osp.join(
        cfg.test_dataloader.dataset.data_root, 'test')

    print('Dataset Building')
    dataset = _build_dataset(hypes, train=False, ego_candidates=ego_candidates)
    print(f'{len(dataset)} samples found.')

    # Match upstream inference.py exactly: batch_size=1, num_workers=16,
    # PyTorch default prefetch. Going higher can thrash networked storage
    # (sshfs / NFS) where I/O is bandwidth-limited rather than throughput
    # bound.
    num_workers = max(int(cfg.test_dataloader.num_workers), 16)
    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers,
        collate_fn=dataset.collate_batch_test, shuffle=False,
        pin_memory=False, drop_last=False)

    print('Creating Model')
    model = _build_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ckpt = args.checkpoint if getattr(args, 'checkpoint', None) \
        else cfg.load_from
    print(f'Loading Model from checkpoint {ckpt}')
    _load_ckpt(model, ckpt, strict=False)
    model.eval()

    def _empty():
        return {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat = _empty()
    result_stat_3d = _empty()

    for batch_data in tqdm(data_loader):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            pred_box, pred_score, gt_box = \
                inference_utils.inference_intermediate_fusion(
                    batch_data, model, dataset)
            for thr in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(
                    pred_box, pred_score, gt_box, result_stat, thr,
                    iou_mode='bev')
                eval_utils.caluclate_tp_fp(
                    pred_box, pred_score, gt_box, result_stat_3d, thr,
                    iou_mode='3d')

    eval_utils.eval_final_results(
        result_stat, work_dir, global_sort_detections=False, label='bev')
    eval_utils.eval_final_results(
        result_stat_3d, work_dir, global_sort_detections=False, label='3d')
