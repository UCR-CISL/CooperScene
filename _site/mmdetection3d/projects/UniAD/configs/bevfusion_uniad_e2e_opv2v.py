"""BEVFusion + UniAD End-to-End config (Plan C).

Architecture:
    BEVFusion lidar encoder (frozen, pretrained)
        → BEV features (B, 512, 200, 200)
        → BEV Adapter (512→256)
            → DETR3D Decoder (900 queries, 6 layers, deformable cross-attention)
                → Detection results + rich 256-d query embeddings
                    → RuntimeTracker (ID assignment)
                    → MemoryBank (4-frame temporal cross-attention)
                    → QIM (query interaction module)
                        → Active track queries
                            → MotionHead (MotionFormer, 3 layers, 6 modes × 12 steps)
                                → Trajectory predictions + losses

Usage:
    python tools/train.py projects/UniAD/configs/bevfusion_uniad_e2e_opv2v.py
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'projects.BEVFusion.bevfusion',
        'projects.UniAD.uniad',
    ],
    allow_failed_imports=False)

# ===================== Shared Settings =====================
voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [-72.0, -72.0, -5.0, 72.0, 72.0, 3.0]
grid_size = [1440, 1440, 41]

class_names = ['vehicle']
num_classes = 1
metainfo = dict(classes=class_names)

bev_h = 100
bev_w = 100
embed_dims = 256

predict_steps = 12
past_steps = 4
num_anchor = 6
# 300 queries (vs UniAD's 900) to fit single RTX 4090 GPU
num_query = 300

# ===================== BEVFusion (lidar-only, frozen) =====================
bevfusion_cfg = dict(
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
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=(
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
            (128, 128),
        ),
        encoder_paddings=(
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, (1, 1, 0)),
            (0, 0),
        ),
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
        upsample_cfg=dict(type='deconv', bias=False),
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        use_conv_for_no_stride=True),
    # TransFusionHead required by BEVFusion init (not used in E2E pipeline)
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
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            post_center_range=[-90.0, -90.0, -8.0, 90.0, 90.0, 8.0],
            score_threshold=0.0,
            code_size=8),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            reduction='mean',
            loss_weight=0.25),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0),
        train_cfg=dict(
            dataset='OPV2V',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='OPV2V',
            grid_size=grid_size,
            out_size_factor=8,
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            nms_type=None)),
)

# ===================== DETR3D Detection Head =====================
detr3d_head_cfg = dict(
    embed_dims=embed_dims,
    num_query=num_query,
    num_classes=num_classes,
    num_decoder_layers=6,
    num_heads=8,
    feedforward_dims=512,
    dropout=0.1,
    num_points=4,
    pc_range=point_cloud_range,
    code_size=10,
    bev_h=bev_h,
    bev_w=bev_w,
    past_steps=past_steps,
    fut_steps=past_steps,
)

# ===================== Tracking Components =====================
qim_cfg = dict(
    embed_dims=embed_dims,
    num_heads=8,
    feedforward_dims=2048,
    dropout=0.0,
    random_drop=0.1,
    fp_ratio=0.3,
    update_query_pos=True,
)

memory_bank_cfg = dict(
    embed_dims=embed_dims,
    num_heads=8,
    feedforward_dims=2048,
    memory_bank_len=4,
    memory_bank_score_thresh=0.0,
    save_period=3,
)

tracker_cfg = dict(
    score_thresh=0.5,
    filter_score_thresh=0.4,
    miss_tolerance=5,
)

# ===================== Model =====================
data_preprocessor_cfg = dict(
    type='QueueDet3DDataPreprocessor',
    pad_size_divisor=32)

model = dict(
    type='BEVFusionUniADE2E',
    data_preprocessor=data_preprocessor_cfg,
    bevfusion=bevfusion_cfg,
    bev_adapter=dict(
        in_channels=512,
        out_channels=embed_dims,
        in_size=200,
        out_size=bev_h,
    ),
    detr3d_head=detr3d_head_cfg,
    qim=qim_cfg,
    memory_bank=memory_bank_cfg,
    tracker=tracker_cfg,
    motion_head=dict(
        type='MotionHead',
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=point_cloud_range,
        embed_dims=embed_dims,
        num_anchor=num_anchor,
        predict_steps=predict_steps,
        num_decoder_layers=3,
        num_heads=8,
        feedforward_dims=512,
        num_points=4,
        dropout=0.1,
        use_bev_interaction=True,
        cls_loss_weight=0.5,
        nll_loss_weight=0.5,
        loss_weight_minade=0.0,
        loss_weight_minfde=0.25,
        use_variance=True,
        anchor_path='projects/UniAD/configs/motion_anchors_k6.npy',
    ),
    map_encoder=dict(
        type='BEVMapEncoder',
        in_channels=6,
        embed_dims=embed_dims,
        num_queries=50,
        map_size=256,
    ),
    planning_head=None,
    freeze_perception=True,
    use_gt_train=True,
    bevfusion_checkpoint='work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d_lrA/epoch_25.pth',
    num_query=num_query,
    num_classes=num_classes,
    pc_range=point_cloud_range,
    embed_dims=embed_dims,
)

# ===================== Dataset =====================
dataset_type = 'OPV2VMotionDataset'
data_root = '/workspace/data/OPV2V/'
data_prefix = dict(pts='')
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
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='LoadBEVMap', map_types=['bev_lane', 'bev_static'], map_size=256),
    dict(
        type='Pack3DDetInputs',
        keys=('points', 'gt_bboxes_3d', 'gt_labels_3d'),
        meta_keys=(
            'box_type_3d', 'sample_idx', 'lidar_path',
            'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans',
            'scenario', 'agent_id', 'timestamp',
            'gt_fut_traj', 'gt_fut_traj_mask',
            'gt_past_traj', 'gt_past_traj_mask',
            'gt_vehicle_ids',
            'bev_map',
        ))
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadBEVMap', map_types=['bev_lane', 'bev_static'], map_size=256),
    dict(
        type='Pack3DDetInputs',
        keys=('points',),
        meta_keys=(
            'box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats',
            'scenario', 'agent_id', 'timestamp',
            'gt_fut_traj', 'gt_fut_traj_mask',
            'gt_past_traj', 'gt_past_traj_mask',
            'bev_map',
        ))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='opv2v_motion_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        pcd_limit_range=point_cloud_range,
        future_steps=predict_steps,
        past_steps=past_steps,
        queue_length=past_steps + 1,  # 5 frames: 4 history + 1 current
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
        ann_file='opv2v_motion_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=point_cloud_range,
        future_steps=predict_steps,
        past_steps=past_steps,
        backend_args=backend_args))

test_dataloader = val_dataloader

# ===================== Evaluator =====================
val_evaluator = [
    dict(
        type='OPV2VMetric',
        ann_file=data_root + 'opv2v_motion_infos_val.pkl',
        metric='bbox',
        iou_thresholds=[0.3, 0.5, 0.7],
        backend_args=backend_args),
    dict(
        type='MotionMetric',
        predict_steps=predict_steps,
        num_modes_eval=[1, 6],
        miss_rate_threshold=2.0),
]
test_evaluator = val_evaluator

# ===================== Visualizer =====================
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='opv2v-uniad',
            name='bevfusion_uniad_e2e_temporal',
        ),
    ),
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ===================== Training =====================
lr = 2e-4
max_epochs = 20

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=lr * 0.01,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    # Freeze BEVFusion perception weights
    paramwise_cfg=dict(
        custom_keys=dict(
            bevfusion=dict(lr_mult=0, decay_mult=0))))

auto_scale_lr = dict(enable=False, base_batch_size=16)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
