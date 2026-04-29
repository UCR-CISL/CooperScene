_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.coop.mmdet3d_plugin'],
    allow_failed_imports=False)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

voxel_size = [0.4, 0.4, 4]
point_cloud_range = [-140.8, -40, -3, 140.8, 40, 1]

model = dict(
    type='CooperativeDetector',
    max_cav=5,
    fusion_type='intermediate',
    data_preprocessor=dict(
        type='CoopDet3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(32000, 70000))),

    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),

    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[200, 704]),

    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 8],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),

    shrink_header=dict(
        type='DownsampleConv',
        input_dim=384,
        dim=[256],
        kernal_size=[3],
        stride=[2],
        padding=[1]),

    compression=dict(
        type='NaiveCompressor', input_dim=256, compress_ratio=32),

    fusion_module=dict(
        type='V2VAttFusion',
        feature_dim=256),

    bbox_head=dict(
        type='DetHead',
        in_channels=256,
        anchor_number=2,
        anchor_size=[3.9, 1.6, 1.56],
        anchor_rotations=[0, 90],
        anchor_z=-1.0,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        feature_stride=4,
        pos_threshold=0.6,
        neg_threshold=0.45,
        score_threshold=0.20,
        nms_threshold=0.15,
        max_num=100,
        cls_weight=1.0,
        reg_weight=2.0),

    train_cfg=None,
    test_cfg=None)

dataset_type = 'OPV2VCoopDataset'
data_root = '/data/OPV2V'

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4),
    dict(type='LoadCooperativePointCloud', coord_type='LIDAR',
         load_dim=4, use_dim=[0, 1, 2, 3], max_cav=5,
         proj_first=True,
         point_cloud_range=point_cloud_range),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PackCooperative3DDetInputs',
         keys=['gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4),
    dict(type='LoadCooperativePointCloud', coord_type='LIDAR',
         load_dim=4, use_dim=[0, 1, 2, 3], max_cav=5,
         proj_first=True,
         point_cloud_range=point_cloud_range),
    dict(type='PackCooperative3DDetInputs', keys=[]),
]

train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='cooperative_collate'),
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_coop_infos_train.pkl',
        data_prefix=dict(pts=''),
        pipeline=train_pipeline,
        max_cav=5,
        com_range=70))

val_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='cooperative_collate'),
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_coop_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        test_mode=True,
        max_cav=5,
        com_range=70))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='OPV2VMetric',
    ann_file=data_root + '/opv2v_coop_infos_val.pkl')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, eps=1e-10, weight_decay=1e-4))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.2, by_epoch=True,
         begin=0, end=10),
    dict(type='CosineAnnealingLR', by_epoch=True,
         begin=10, end=90, eta_min=2e-5),
]

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10))
