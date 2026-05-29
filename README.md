# CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

[**arXiv**](TBD) &nbsp;|&nbsp; [**Project Website**](TBD) &nbsp;|&nbsp; [**Full demo video (mp4)**](assets/take_10.mp4)

![Demo](assets/take_10.gif)

CooperScene is the first real-world, multi-agent, multi-modal cooperative autonomy
dataset with C-V2X communication characterization. It features three connected
autonomous vehicles (CAVs) and one instrumented infrastructure roadside unit (RSU),
all equipped with multi-modal sensors and commercial off-the-shelf C-V2X
communication radios, interacting across diverse real-world traffic scenarios
including intersections, highway ramps, and parking areas.

**Key highlights**

- **59K** synchronized LiDAR frames across 4 cooperative agents
- **53K** camera image frames
- **344K** globally consistent 3D bounding box labels at 10 Hz
- Real-world C-V2X communication traces (latency, throughput, packet loss, jitter)
- Centimeter-level localization via GNSS-RTK + spatial-temporal ICP alignment
- Sub-millisecond sensor synchronization via PTP and hardware triggering

---

## Quick Start

### Docker

Build a local image (extends a pre-built base with all Python + CUDA dependencies
— PyTorch 2.1.1, mmdet3d 1.4, mmcv 2.1, spconv 2.x — and adds an entrypoint
that auto-compiles the BEVFusion CUDA ops on first run):

```bash
docker build -t cooperscene:latest -f docker/Dockerfile docker/

docker run --gpus all -it --rm \
    -v $(pwd):/workspace/CooperScene \
    -v /path/to/data:/data \
    cooperscene:latest \
    bash
```

The entrypoint runs `python setup.py develop` under `models/bevfusion/` if
`bev_pool_ext*.so` is not already present, then exec's your command. Subsequent
runs skip the build. The other cooperative configs (`configs/cobevt/`,
`configs/v2vam/`, `configs/v2vnet/`, `configs/v2xvit/`) do **not** use these
ops, so the build is only needed for BEVFusion configs.

If you prefer the bare base image:

```bash
docker pull bwu109/motion_prediction@sha256:32e06e6533ce82d267696b8821b9f494d2f508971ab5501e736a65f1fb1ddcc3
# then from the project root (NOT from models/bevfusion/):
python models/bevfusion/setup.py develop --user
```

`--user` is required if site-packages is read-only (Apptainer, locked-down
Docker images). The compiled `.so` lands in the source tree either way.
The setup.py uses sources relative to the project root, so it must be invoked
from the root rather than from `models/bevfusion/` (running it inside that
directory produces a duplicated path and ninja fails).

---

## Data Download & Preparation

### Download

*TBD* — public release link will be posted at
<https://data.ucr.edu/datasets/cooperscene/>.

The benchmark ships in **OPV2V format**:
`<split>/<take>/<agent>/<frame>.{pcd,yaml}` plus `<frame>_camera0.png` on
camera-equipped agents. Each take has **4 agents**: agent `0` is LiDAR-only;
agents `1–3` also carry a front camera.

```
cooperscene/
├── train/
│   ├── 1/                    # take id
│   │   ├── 0/                # agent 0 — LiDAR only
│   │   │   ├── 481260.pcd
│   │   │   ├── 481260.yaml   # pose + GT bboxes
│   │   │   └── ...
│   │   ├── 1/                # agent 1 — LiDAR + front camera
│   │   │   ├── 481260.pcd
│   │   │   ├── 481260.yaml
│   │   │   └── 481260_camera0.png
│   │   ├── 2/  3/            # other agents — same layout as agent 1
│   ├── 2/  3/  ...           # other takes — same 4-agent layout
├── validate/
├── test/
└── mcap/                     # per-take MCAP recordings (LiDAR + camera + throughput)
    ├── 1.mcap
    ├── 2.mcap
    └── ...
```

A **mini set** of 180 contiguous frames (120 train / 30 validate / 30 test) 
is shipped alongside the full release for pipeline smoke tests — same `<split>/<take>/<agent>/<frame>` layout, rooted at
`/mini`.

### Generate `.pkl` index files (and `.bin` point clouds)

mmdet3d-style `.pkl` info files are generated from raw OPV2V/CooperScene scenes
by the converters under `tools/dataset_converters/`.

**Cooperative models** (CoBEVT / V2VAM / V2VNet / V2X-ViT / coop BEVFusion):

```bash
python tools/dataset_converters/opv2v_cooper_converter.py \
    --data-root /workspace/cooperscene_mini \
    --out-dir   /workspace/cooperscene_mini \
    --convert-pcd
```

**Single-agent BEVFusion** (lidar / lidar-cam):

```bash
python tools/dataset_converters/opv2v_converter.py \
    --data-root /workspace/cooperscene_mini \
    --out-dir   /workspace/cooperscene_mini \
    --convert-pcd
```

`--data-root` must contain `train/`, `validate/`, `test/`. Set `--out-dir`
equal to `--data-root` so configs find the `.pkl` files. Pass `--convert-pcd`
on the first run (writes `.bin` next to each `.pcd`); drop it on later runs.

---

## Training & Inference

### Configs

All configs live under `configs/`:

```
configs/
├── bevfusion/      # BEVFusion (single-agent + cooperative)
├── cobevt/cobevt.py
├── v2vam/v2vam.py
├── v2vnet/v2vnet.py
└── v2xvit/v2xvit.py
```

Each cooperative-perception config (`cobevt/v2vam/v2vnet/v2xvit`) is
self-contained and shares the same field layout. Common knobs you'll
override most often:

| Field | Meaning |
|---|---|
| `train_dataloader.batch_size` | per-GPU batch (default 4) |
| `train_dataloader.num_workers` | worker processes (default 4) |
| `train_dataloader.dataset.data_root` | dataset root containing `train/`, `validate/`, `test/` |
| `optim_wrapper.optimizer.lr` | base learning rate (default `1e-3`) |
| `train_cfg.max_epochs` | total epochs (default 60) |
| `ego_candidates` | rotate ego across these agent IDs per (scenario, timestamp). Default `None` = use the agent with the smallest `cav_id` |
| `load_from` | warm-start ckpt path |

BEVFusion configs (mmengine path) additionally use
`train_dataloader.dataset.ann_file` for the `.pkl` index.

Override any of these from the CLI with `--cfg-options`:

```bash
python tools/train.py configs/v2vam/v2vam.py \
    --cfg-options \
        train_dataloader.batch_size=2 \
        train_dataloader.num_workers=8 \
        train_dataloader.dataset.data_root=$DR \
        val_dataloader.dataset.data_root=$DR \
        test_dataloader.dataset.data_root=$DR \
        train_cfg.max_epochs=40 \
        optim_wrapper.optimizer.lr=5e-4 \
        "ego_candidates=['1','2','3']" \
        load_from=work_dirs/opencood_converted/v2vam.pth
```

### Train

```bash
python tools/train.py configs/cobevt/cobevt.py
```

Swap in any other config under `configs/`. Checkpoints and logs land under
`work_dirs/<config_stem>/`.

### Inference

```bash
python tools/test.py configs/cobevt/cobevt.py \
    work_dirs/cobevt/epoch_60.pth
```

### Pre-trained Checkpoints


<https://drive.google.com/drive/folders/129KNKz9ovrBB_DZ-NS9MUTai2p9PG1cv?usp=sharing>

use `load_from=...` to warm-start training or `tools/test.py <cfg> <ckpt>` for inference. 

---

## Benchmark Result

*TBD* 

---

## Citation

TBD

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) — cooperative perception framework
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) — 3D detection toolbox
- [BEVFusion](https://github.com/mit-han-lab/bevfusion) — multi-modal BEV fusion
