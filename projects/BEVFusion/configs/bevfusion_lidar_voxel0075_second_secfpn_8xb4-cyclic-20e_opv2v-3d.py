_base_ = ['../../coop/configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)

# ===================== Model Settings =====================
# Voxel size for voxel encoder
# OPV2V has a different point cloud range than nuScenes
# Adjust based on your OPV2V data characteristics
voxel_size = [0.1, 0.1, 0.2]
# OPV2V typically has a larger range since it's from CARLA simulation
# X/Y range set to [-72, 72] so grid_size=1440, which after /8 = 180 (even number)
# Z range set to [-5, 3] to get sparse_shape[2]=41 (same as nuScenes)
# This ensures the sparse encoder outputs 256 channels (128 * D where D=2)
point_cloud_range = [-72.0, -72.0, -5.0, 72.0, 72.0, 3.0]

# OPV2V only has vehicle class
class_names = ['vehicle']
num_classes = 1

# Calculate grid size from point cloud range and voxel size
# grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
# For [-72, -72, -5, 72, 72, 3] with voxel [0.1, 0.1, 0.2]:
# grid_size = [1440, 1440, 41]
# 1440 / 8 = 180 (even number, avoids FPN size mismatch)
grid_size = [1440, 1440, 41]

metainfo = dict(classes=class_names)
dataset_type = 'OPV2VDataset'
data_root = '/workspace/data/OPV2V/'
data_prefix = dict(pts='')  # Point clouds are at data_root/split/scenario/agent/

input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000],
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),  # x,y,z,intensity
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=num_classes,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='OPV2V',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            # OPV2V: 7 dims (x,y,z,dx,dy,dz,yaw), no velocity
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='OPV2V',
            grid_size=grid_size,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None),
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2]),  # No velocity for OPV2V
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-82.0, -82.0, -8.0, 82.0, 82.0, 8.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=8),  # 8 dims without velocity
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

# ===================== Dataset Settings =====================
# Use LoadPointsFromPCD for .pcd files, or LoadPointsFromFile for .bin
# If you converted .pcd to .bin, use LoadPointsFromFile
train_pipeline = [
    dict(
        type='LoadPointsFromFile',  # Change to LoadPointsFromPCD if using .pcd
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path',
            'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans'
        ])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',  # Change to LoadPointsFromPCD if using .pcd
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        pcd_limit_range=[-72.0, -72.0, -5.0, 72.0, 72.0, 3.0],
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=[-72.0, -72.0, -5.0, 72.0, 72.0, 3.0],
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_infos_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=[-72.0, -72.0, -5.0, 72.0, 72.0, 3.0],
        backend_args=backend_args))

# ===================== Evaluator Settings =====================
# OPV2V metric computes 3D mAP at different IoU thresholds
val_evaluator = dict(
    type='OPV2VMetric',
    ann_file=data_root + 'opv2v_infos_val.pkl',
    metric='bbox',
    iou_thresholds=[0.3, 0.5, 0.7],
    backend_args=backend_args)
test_evaluator = dict(
    type='OPV2VMetric',
    ann_file=data_root + 'opv2v_infos_test.pkl',
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
            name='bevfusion_lidar_opv2v',
        ),
    ),
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ===================== Training Settings =====================
lr = 0.0001
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=32)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
