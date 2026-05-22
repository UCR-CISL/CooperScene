_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d_lrA.py'
]

custom_imports = dict(
    imports=['models.bevfusion'], allow_failed_imports=False)

# ===================== Cooperative LiDAR+Camera Model =====================
# Per vehicle (each in OWN coordinate frame):
#   LiDAR -> Voxelize -> SparseEncoder -> lidar_BEV (256ch)
#   Cameras -> Swin -> FPN -> DepthLSS -> cam_BEV (80ch)
#   ConvFuser(cam_BEV, lidar_BEV) -> fused_BEV (256ch)
#
# STTF warp cooperator fused_BEV -> ego frame
# All fused_BEV -> Backbone -> Neck -> (512ch)
# compress(512->256) -> SwapFusion -> expand(256->512) -> TransFusionHead

point_cloud_range = [-72.0, -72.0, -5.0, 72.0, 72.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

max_cav = 3

model = dict(
    type='CoopBEVFusionLidarCam',
    data_preprocessor=dict(
        type='CoopDet3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    # Camera branch: Swin-T + GeneralizedLSSFPN + DepthLSS + ConvFuser
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=True,  # gradient checkpointing for memory savings
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 512],
        feature_size=[32, 64],
        xbound=[-72.0, 72.0, 0.4],
        ybound=[-72.0, 72.0, 0.4],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256),
    # Cooperative fusion components
    max_cav=max_cav,
    # STTF: 144m range / 180px = 0.8 m/px
    sttf=dict(discrete_ratio=0.8, downsample_rate=1),
    fusion_channels=256,  # compress 512->256 before SwapFusion
    coop_fusion=dict(
        type='SwapFusionEncoder',
        channels=256,
        n_head=8,
        n_layers=3,
        window_size=9,
        agent_size=max_cav,
        mlp_dim=256,
        dim_head=32,
        dropout=0.1,
        use_mask=True),
)

# ===================== Dataset Settings =====================
dataset_type = 'OPV2VCoopDataset'
data_root = '/workspace/data/OPV2V/'

class_names = ['vehicle']
metainfo = dict(classes=class_names)

train_pipeline = [
    # Ego camera images (standard pipeline)
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    # Ego LiDAR
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # Cooperator LiDAR (proj_first=False: keep in own frame for STTF)
    dict(
        type='LoadCoopPointsFromFile',
        load_dim=4,
        use_dim=4,
        proj_first=False),
    # Cooperator cameras
    dict(
        type='LoadCoopCameraData',
        to_float32=True,
        color_type='color',
        apply_aug=True,
        final_dim=[256, 512],
        resize_lim=[0.64, 0.80],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    # Ego camera augmentation
    dict(
        type='ImageAug3D',
        final_dim=[256, 512],
        resize_lim=[0.64, 0.80],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    # No 3D augmentation (BEVFusionGlobalRotScaleTrans, BEVFusionRandomFlip3D)
    # to avoid ego/coop frame misalignment
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=30,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='CoopPack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
              'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats',
            'coop_mask',
        ]),
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
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
        proj_first=False),
    dict(
        type='LoadCoopCameraData',
        to_float32=True,
        color_type='color',
        apply_aug=True,
        final_dim=[256, 512],
        resize_lim=[0.72, 0.72],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 512],
        resize_lim=[0.72, 0.72],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='CoopPack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats',
            'coop_mask',
        ]),
]

data_prefix = dict(pts='', img='')

train_dataloader = dict(
    batch_size=2,
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
    batch_size=2,
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
    batch_size=2,
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
    type='OPV2VMetric',
    ann_file=data_root + 'opv2v_coop_infos_val.pkl',
    metric='bbox',
    iou_thresholds=[0.3, 0.5, 0.7],
    backend_args=backend_args)
test_evaluator = dict(
    type='OPV2VMetric',
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
            name='bevfusion_coop_lidarcam',
        ),
    ),
]
# vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ===================== Training Settings =====================
# Fine-tuning from merged checkpoint
lr = 0.0001
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=lr * 1e-4,
        begin=0,
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
    # Freeze LiDAR encoder, slow LR for camera backbone
    paramwise_cfg=dict(
        custom_keys={
            'pts_voxel_layer': dict(lr_mult=0, decay_mult=0),
            'pts_voxel_encoder': dict(lr_mult=0, decay_mult=0),
            'pts_middle_encoder': dict(lr_mult=0, decay_mult=0),
            'img_backbone': dict(lr_mult=0.1),
        }))

auto_scale_lr = dict(enable=False, base_batch_size=16)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=2))

# Load merged checkpoint for fine-tuning
# Set via command line: --cfg-options load_from='path/to/merged_ckpt.pth'
load_from = None
