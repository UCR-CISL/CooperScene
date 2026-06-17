# BEVFusion for CooperScene — Cooperative 3D Object Detection

BEVFusion support for the **CooperScene** cooperative perception dataset,
extending [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)'s BEVFusion.

## Environment

See the top-level [README](../../README.md) for the Docker image (built from
scratch) or the local install. The BEVFusion CUDA ops are compiled with:

```bash
# from the project root (not from inside models/bevfusion/)
python models/bevfusion/setup.py develop --user
```

---

## Data Preparation

### Expected directory layout

```
data/cooperscene/
├── train/
│   └── <take>/
│       └── <agent>/
│           ├── <timestamp>.pcd (or .bin)
│           ├── <timestamp>.yaml
│           └── <timestamp>_camera0.png
├── validate/
├── test/
├── cooperscene_infos_train.pkl            # generated (single-agent)
└── cooperscene_coop_infos_train.pkl       # generated (cooperative)
```

### Generate info files

Single-agent (lidar / lidar-cam):

```bash
python tools/dataset_converters/data_converter.py \
    --data-root data/cooperscene \
    --out-dir   data/cooperscene \
    --splits train validate test \
    --convert-pcd          # optional: convert .pcd to .bin
```

Cooperative:

```bash
python tools/dataset_converters/cooperscene_converter.py \
    --data-root data/cooperscene \
    --convert-pcd
```

---

## Training

Each stage can warm-start from the previous one via
`--cfg-options load_from=<CHECKPOINT>` (a pretrained checkpoint also works as a
starting point):

```bash
# Stage 1 — single-agent LiDAR
python tools/train.py configs/bevfusion/bevfusion_lidar_cooperscene.py

# Stage 2 — single-agent LiDAR + camera (warm-start from stage 1)
python tools/train.py configs/bevfusion/bevfusion_lidar-cam_cooperscene.py \
    --cfg-options load_from=<LIDAR_CHECKPOINT> \
                  model.img_backbone.init_cfg.checkpoint=<SWIN_PRETRAINED>

# Stage 3 — cooperative LiDAR (warm-start from stage 1)
python tools/train.py configs/bevfusion/bevfusion_coop_lidar_cooperscene.py \
    --cfg-options load_from=<SINGLE_AGENT_LIDAR_CHECKPOINT>

# Stage 4 — cooperative LiDAR + camera
python tools/train.py configs/bevfusion/bevfusion_coop_lidarcam_cooperscene.py \
    --cfg-options load_from=<LIDARCAM_OR_COOP_LIDAR_CHECKPOINT>
```

For 5 CAVs, use the `*_cooperscene_5cav.py` variants. Multi-GPU:

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
