"""BEVFusion + UniAD End-to-End config — multi-frame temporal queue.

Same architecture as bevfusion_uniad_e2e_opv2v.py, but with queue_length=4
for true temporal training matching original UniAD:
    frame t-3 → frame t-2 → frame t-1 → frame t
    Detection loss on ALL frames, motion loss only on frame t.
    MemoryBank/QIM get gradients through the temporal chain.

Requires batch_size=1 (one temporal queue per GPU).

Usage:
    python tools/train.py projects/UniAD/configs/bevfusion_uniad_e2e_opv2v_queue.py
"""

_base_ = ['bevfusion_uniad_e2e_opv2v.py']

queue_length = 4

# Queue-aware preprocessor
data_preprocessor_cfg = dict(
    type='QueueDet3DDataPreprocessor',
    pad_size_divisor=32)

model = dict(
    data_preprocessor=data_preprocessor_cfg,
    use_tracking=True,
    queue_length=queue_length,
)

# Batch_size=1 required for temporal queue training
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(queue_length=queue_length))

# Lower LR for smaller effective batch_size
lr = 1e-4
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=lr * 0.01,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Different wandb run name
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='opv2v-uniad',
            name='bevfusion_uniad_e2e_queue4',
        ),
    ),
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
