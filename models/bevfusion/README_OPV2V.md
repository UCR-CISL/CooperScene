# BEVFusion for OPV2V & CooperScene — Cooperative 3D Object Detection

This fork extends [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)'s BEVFusion with support for the **OPV2V** and **CooperScene** cooperative perception datasets.

## Docker

Pre-built environment with all dependencies (See available tags at: <https://hub.docker.com/repository/docker/bwu109/motion_prediction/tags>):

```bash
docker pull bwu109/motion_prediction:<tag>
```

---

## Setup

Compile CUDA ops:

```bash
python projects/BEVFusion/setup.py develop
```

---

## Data Preparation

### Expected Directory Layout

**OPV2V:**
```
data/opv2v/
├── train/
│   └── <scenario>/
│       └── <agent_id>/
│           ├── <timestamp>.pcd (or .bin)
│           ├── <timestamp>.yaml
│           └── camera0..3/
│               └── <timestamp>.png
├── validate/
├── test/
├── opv2v_infos_train.pkl                 # generated
├── opv2v_coop_infos_train.pkl            # generated (cooperative)
└── ...
```

same for CooperScene

### Generate Info Files

**OPV2V (single-agent):**
```bash
python tools/dataset_converters/opv2v_converter.py \
    --data-root data/opv2v \
    --out-dir data/opv2v \
    --splits train validate test \
    --convert-pcd          # optional: convert .pcd to .bin
```

**OPV2V (cooperative):**
```bash
python tools/dataset_converters/opv2v_cooper_converter.py \
    --data-root data/opv2v \
    --out-dir data/opv2v \
    --splits train validate test
```


---

## Training

### Training Pipeline

```
OPV2V LiDAR (single-agent)
    │
    ├──► OPV2V LiDAR+Camera (single-agent, stage 2)
    │
    ├──► Coop LiDAR
    │
    ├──► Coop LiDAR+Camera  (load from single-agent LiDAR+Camera or Coop LiDAR)
    │
    └──► CooperScene LiDAR (transfer)
            │
            └──► CooperScene LiDAR+Camera (stage 2)
```

### Step 1: Single-Agent LiDAR (OPV2V)

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d_lrA.py
```

### Step 2: Single-Agent LiDAR+Camera (OPV2V)

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d.py \
    --cfg-options load_from=<LIDAR_CHECKPOINT> \
                  model.img_backbone.init_cfg.checkpoint=<SWIN_PRETRAINED>
```

### Step 3: Cooperative LiDAR (OPV2V)

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_coop_lidar_opv2v.py \
    --cfg-options load_from=<SINGLE_AGENT_LIDAR_CHECKPOINT>
```

For 5 CAVs, use `bevfusion_coop_lidar_opv2v_5cav.py` instead.

### Step 4: Cooperative LiDAR+Camera (OPV2V)

Load from either the single-agent LiDAR+Camera checkpoint or the cooperative LiDAR checkpoint:

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_coop_lidarcam_opv2v.py \
    --cfg-options load_from=<LIDARCAM_OR_COOP_LIDAR_CHECKPOINT>
```

For 5 CAVs, use `bevfusion_coop_lidarcam_opv2v_5cav.py` instead.

### Step 5: CooperScene (transfer from OPV2V)

```bash
# LiDAR
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar_cooperscene.py \
    --cfg-options load_from=<OPV2V_LIDAR_CHECKPOINT>

# LiDAR+Camera (stage 2)
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_cooperscene.py \
    --cfg-options load_from=<COOPERSCENE_LIDAR_CHECKPOINT>
```

### Multi-GPU Training

```bash
bash tools/dist_train.sh <CONFIG> <NUM_GPUS>
```

---

## Testing

```bash
python tools/test.py <CONFIG> <CHECKPOINT>

# Multi-GPU
bash tools/dist_test.sh <CONFIG> <CHECKPOINT> <NUM_GPUS>
```

Metrics reported: `AP_BEV@0.3/0.5/0.7`, `AP_3D@0.3/0.5/0.7`, `mAP_BEV`, `mAP_3D`.

---
