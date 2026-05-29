_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d.py'
]

custom_imports = dict(
    imports=['models.bevfusion'], allow_failed_imports=False)

# ===================== Cooperative Model Settings =====================
# Cooperative BEVFusion: multi-agent LiDAR fusion with CoBEVT SwapFusion
# Fine-tune from single-agent checkpoint

# Point cloud range and voxel size inherited from base
# proj_first: cooperator points are transformed to ego frame before
# feature extraction, so no STTF BEV warp is needed.

max_cav = 5

model = dict(
    type='CoopBEVFusion',
    data_preprocessor=dict(
        type='CoopDet3DDataPreprocessor'),
    max_cav=max_cav,
    fusion_channels=256,  # compress 512->256 before fusion, expand back
    coop_fusion=dict(
        type='SwapFusionEncoder',
        channels=256,
        n_head=8,
        n_layers=3,
        window_size=9,   # 180 / 9 = 20 windows
        agent_size=max_cav,
        mlp_dim=256,
        dim_head=32,
        dropout=0.1,
        use_mask=True),
)

# ===================== Dataset Settings =====================
dataset_type = 'CoopDataset'
data_root = '/workspace/data/OPV2V/'

class_names = ['vehicle']
metainfo = dict(classes=class_names)
point_cloud_range = [-72.0, -72.0, -5.0, 72.0, 72.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadCoopPointsFromFile',
        load_dim=4,
        use_dim=4,
        proj_first=True,
        point_cloud_range=point_cloud_range),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='CoopPack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path',
            'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans',
            'coop_mask',
        ]),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadCoopPointsFromFile',
        load_dim=4,
        use_dim=4,
        proj_first=True,
        point_cloud_range=point_cloud_range),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='CoopPack3DDetInputs',
        keys=['points'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats',
            'coop_mask',
        ]),
]

data_prefix = dict(pts='')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_coop_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        max_cav=max_cav,
        com_range=70.0,
        pcd_limit_range=[-72.0, -72.0, -5.0, 72.0, 72.0, 3.0],
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_coop_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        max_cav=max_cav,
        com_range=70.0,
        pcd_limit_range=[-72.0, -72.0, -5.0, 72.0, 72.0, 3.0],
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_coop_infos_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        max_cav=max_cav,
        com_range=70.0,
        pcd_limit_range=[-72.0, -72.0, -5.0, 72.0, 72.0, 3.0],
        backend_args=backend_args))

# ===================== Evaluator Settings =====================
val_evaluator = dict(
    type='EvalMetric',
    ann_file=data_root + 'opv2v_coop_infos_val.pkl',
    metric='bbox',
    iou_thresholds=[0.3, 0.5, 0.7],
    backend_args=backend_args)
test_evaluator = dict(
    type='EvalMetric',
    ann_file=data_root + 'opv2v_coop_infos_test.pkl',
    metric='bbox',
    iou_thresholds=[0.3, 0.5, 0.7],
    backend_args=backend_args)

# ===================== Visualizer Settings =====================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='opv2v-bevfusion',
            name='bevfusion_coop_lidar_5cav',
        ),
    ),
]
# vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ===================== Training Settings =====================
# Fine-tuning schedule: shorter than single-agent training
# Freeze backbone initially, train only cooperative fusion + head
lr = 0.0001
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=lr * 2,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=lr * 1e-4,
        begin=10,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=10,
        eta_min=0.85 / 0.95,
        begin=0,
        end=10,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=20,
        eta_min=1,
        begin=10,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    # Freeze pre-trained backbone for cooperative fine-tuning
    paramwise_cfg=dict(
        custom_keys={
            'pts_voxel_layer': dict(lr_mult=0, decay_mult=0),
            'pts_voxel_encoder': dict(lr_mult=0, decay_mult=0),
            'pts_middle_encoder': dict(lr_mult=0, decay_mult=0),
        }))

auto_scale_lr = dict(enable=False, base_batch_size=16)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=2))

# Load single-agent checkpoint for fine-tuning
# Set via command line: --cfg-options load_from='path/to/checkpoint.pth'
load_from = None
