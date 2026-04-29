# Training & Testing

This page covers two benchmark families:

1. **Cooperative 3D Detection** (CoBEVT, V2VAM, V2VNet, V2X-ViT) under
   `projects/coop/`
2. **Multi-Modal Cooperative Perception (BEVFusion)** under
   `projects/BEVFusion/` — single-agent, cooperative, OPV2V → CooperScene
   transfer

> Run everything from the repo root with the project on `PYTHONPATH`:
> ```bash
> cd CooperScene
> export PYTHONPATH=$(pwd):$PYTHONPATH
> ```

---

## 1. Cooperative 3D Detection

Four configs live in `projects/coop/configs/opv2v/`:

| Model | Config | Epochs | LR schedule |
|---|---|---|---|
| **CoBEVT** | `pointpillars_cobevt.py` | 90 | cosineannealwarm, warmup 10 |
| **V2VAM** | `pointpillars_v2vam.py` | 60 | cosineannealwarm, warmup 10 |
| **V2VNet** | `pointpillars_v2vnet.py` | 150 | multistep `[80, 100]`, γ=0.1 |
| **V2X-ViT** | `pointpillars_v2xvit.py` | 60 | multistep `[15, 65]`, γ=0.1 |

All four:
- PointPillars backbone + DownsampleConv shrink + (optional) NaiveCompressor
- OpenCOOD-style anchor head (focal cls + smooth-L1 reg with sin-difference)
- Eval metric: `AP_BEV / AP_3D` at IoU thresholds 0.3 / 0.5 / 0.7
- Hyperparameters mirror OpenCOOD's `point_pillar_*.yaml` exactly (voxel size,
  PCR, anchor sizes, loss weights, etc.). See the per-config docstring for any
  deliberate divergence (e.g. V2VAM uses lr=1e-3, not the yaml's 1e-7 finetune
  residue).

### Train one model

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py projects/coop/configs/opv2v/pointpillars_cobevt.py
```

Substitute the config name for the other three. Each config sets its own
`work_dir` under `work_dirs/<config_stem>/`. Checkpoints are saved every 10
epochs (matches OpenCOOD's `save_freq`).

### Train in the background (long runs)

```bash
mkdir -p work_dirs
CUDA_VISIBLE_DEVICES=0 \
nohup python -u tools/train.py \
    projects/coop/configs/opv2v/pointpillars_cobevt.py \
    > work_dirs/cobevt_train.log 2>&1 &
echo "PID=$!"
```

Tail `work_dirs/<config_stem>/<run_id>/<run_id>.log` for the actual mmengine
log (the nohup file mostly captures stderr).

### Multi-GPU

```bash
bash tools/dist_train.sh \
    projects/coop/configs/opv2v/pointpillars_cobevt.py 4
```

(`dist_train.sh` is the standard mmdet3d wrapper around `torch.distributed.run`.)

### Test

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    projects/coop/configs/opv2v/pointpillars_cobevt.py \
    work_dirs/pointpillars_cobevt/epoch_90.pth
```

Outputs:
```
AP_BEV@0.3 / @0.5 / @0.7
AP_3D@0.3  / @0.5  / @0.7
mAP_BEV    , mAP_3D
```

---

## 2. Multi-Modal Cooperative Perception (BEVFusion)

### Pipeline overview

BEVFusion benefits from staged training. The recommended order:

```
OPV2V LiDAR (single-agent, Step 1)
    │
    ├──► OPV2V LiDAR+Camera (single-agent, Step 2)
    │
    ├──► OPV2V Cooperative LiDAR (Step 3, loads from Step 1)
    │
    ├──► OPV2V Cooperative LiDAR+Camera (Step 4, loads from Step 2 or Step 3)
    │
    └──► CooperScene (Step 5, transfer from OPV2V)
            │
            └──► CooperScene LiDAR+Camera (Step 5b)
```

Each step warm-starts from the previous one via `--cfg-options load_from=...`.

### Step 1 — Single-Agent LiDAR (OPV2V)

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d_lrA.py
```

### Step 2 — Single-Agent LiDAR+Camera (OPV2V)

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-30e_opv2v-3d.py \
    --cfg-options load_from=<STEP_1_CHECKPOINT> \
                  model.img_backbone.init_cfg.checkpoint=<SWIN_PRETRAINED>
```

### Step 3 — Cooperative LiDAR (OPV2V)

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_coop_lidar_opv2v.py \
    --cfg-options load_from=<STEP_1_CHECKPOINT>
```

For 5-CAV experiments, use `bevfusion_coop_lidar_opv2v_5cav.py` instead.

### Step 4 — Cooperative LiDAR+Camera (OPV2V)

Load from either the single-agent LiDAR+Camera (Step 2) or the cooperative
LiDAR (Step 3) checkpoint:

```bash
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_coop_lidarcam_opv2v.py \
    --cfg-options load_from=<STEP_2_OR_STEP_3_CHECKPOINT>
```

5-CAV: `bevfusion_coop_lidarcam_opv2v_5cav.py`.

### Step 5 — CooperScene (transfer from OPV2V)

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

### Multi-GPU

```bash
bash tools/dist_train.sh <CONFIG> <NUM_GPUS>
```

### Test

```bash
python tools/test.py <CONFIG> <CHECKPOINT>

# Multi-GPU
bash tools/dist_test.sh <CONFIG> <CHECKPOINT> <NUM_GPUS>
```

Same metrics as the cooperative configs:
`AP_BEV@{0.3,0.5,0.7}`, `AP_3D@{0.3,0.5,0.7}`, `mAP_BEV`, `mAP_3D`.

---

## Common Overrides

All can be passed via `--cfg-options key=value` (no edit needed):

| Override | Example |
|---|---|
| Batch size | `--cfg-options train_dataloader.batch_size=2 val_dataloader.batch_size=2` |
| Mixed precision (AMP) | `--cfg-options optim_wrapper.type=AmpOptimWrapper optim_wrapper.loss_scale=dynamic` |
| Different `data_root` | `--cfg-options data_root=/scratch/OPV2V` |
| Resume training | `--resume` (auto-finds latest ckpt) or `--resume work_dirs/.../epoch_30.pth` |
| Different work dir | `--work-dir work_dirs/cobevt_amp_run3` |

---

## Troubleshooting

- **`bev_pool_ext` ImportError** — BEVFusion CUDA ops not built. Run
  `cd projects/BEVFusion && python setup.py develop`.
- **Out of memory on cooperative configs** — CoBEVT and V2X-ViT use the
  full-resolution feature map (`shrink stride=1`, feature map ~96×352);
  default `batch_size=4` may not fit on 24 GB GPUs. Drop to `batch_size=2`
  (matches OpenCOOD's CoBEVT yaml) or enable AMP (above).
- **Hangs at "Checkpoints will be saved to ..."** — usually a silent OOM in a
  DataLoader worker. Re-run with `--cfg-options train_dataloader.num_workers=0`
  to surface the real exception, or check `nvidia-smi` for a zombie process
  hogging the GPU.
- **`OPV2VCoopDataset is already registered`** — a forked `mmdet3d` is shadowing
  the pip install. Run `pip uninstall mmdet3d && pip install "mmdet3d>=1.4,<1.5"`
  and ensure no `mmdetection3d/` is on your `PYTHONPATH`.
