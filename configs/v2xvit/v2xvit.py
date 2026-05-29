_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(
    imports=['models.cooperative'],
    allow_failed_imports=False)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

voxel_size = [0.4, 0.4, 4]
point_cloud_range = [-140.8, -38.4, -3, 140.8, 38.4, 1]

opencood_args = dict(
    max_cav=5,
    lidar_range=point_cloud_range,
    voxel_size=voxel_size,
    anchor_number=2,
    backbone_fix=False,
    compression=32,
    pillar_vfe=dict(
        num_filters=[64],
        use_absolute_xyz=True,
        use_norm=True,
        with_distance=False),
    point_pillar_scatter=dict(
        num_features=64,
        grid_size=[704, 192, 1]),
    base_bev_backbone=dict(
        layer_nums=[3, 5, 8],
        layer_strides=[2, 2, 2],
        num_filters=[64, 128, 256],
        num_upsample_filter=[128, 128, 128],
        upsample_strides=[1, 2, 4]),
    shrink_header=dict(
        kernal_size=[3],
        stride=[2],
        padding=[1],
        dim=[256],
        input_dim=384),
    transformer=dict(
        encoder=dict(
            cav_att_config=dict(
                RTE_ratio=2, dim=256, dim_head=32, dropout=0.3,
                heads=8, use_RTE=True, use_hetero=True),
            depth=3,
            use_RTE=True,
            RTE_ratio=2,
            feed_forward=dict(dropout=0.3, mlp_dim=256),
            num_blocks=1,
            pwindow_att_config=dict(
                dim=256, dim_head=[16, 32, 64], dropout=0.3,
                fusion_method='split_attn',
                heads=[16, 8, 4],
                relative_pos_embedding=True,
                window_size=[4, 8, 16]),
            sttf=dict(downsample_rate=4, voxel_size=voxel_size),
            use_roi_mask=True)),
)

opencood_anchor_args = dict(
    D=1,
    H=192,
    W=704,
    l=3.9,
    w=1.6,
    h=1.56,
    num=2,
    r=[0, 90],
    cav_lidar_range=point_cloud_range,
    feature_stride=4,
    vd=4,
    vh=0.4,
    vw=0.4,
)

opencood_postprocess_args = dict(
    max_num=100,
    nms_thresh=0.15,
    target_args=dict(
        pos_threshold=0.6,
        neg_threshold=0.45,
        score_threshold=0.20,
    ),
)

opencood_loss_args = dict(
    cls_weight=1.0,
    reg=2.0,
)

model = dict(
    type='OpenCOODCooperativeDetector',
    arch='v2xvit',
    max_cav=5,
    opencood_args=opencood_args,
    anchor_args=opencood_anchor_args,
    postprocess_args=opencood_postprocess_args,
    loss_args=opencood_loss_args,
    data_preprocessor=dict(
        type='OpenCOODCoopDet3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(32000, 70000)),
        cav_lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        max_points_per_voxel=32,
        max_voxel_train=32000,
        max_voxel_test=70000),
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

dataset_type = 'CoopDataset'
data_root = 'data/cooperscene'

train_pipeline = [
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
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        test_mode=True,
        max_cav=5,
        com_range=70))

test_dataloader = val_dataloader

val_evaluator = dict(type='EvalMetric')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, eps=1e-10, weight_decay=1e-4))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.2, by_epoch=True, begin=0, end=10),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=10, end=90,
         eta_min=2e-5),
]

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
