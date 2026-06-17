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

The image builds **from scratch** on the public
`pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel` base and installs the full stack
(mmengine / mmcv 2.1 / mmdet 3.x / mmdet3d 1.4 / spconv). The entrypoint
auto-compiles the BEVFusion CUDA ops on first run.

```bash
docker build -t cooperscene:latest -f docker/Dockerfile docker/

docker run --gpus all -it --rm \
    -v $(pwd):/workspace/CooperScene \
    -v /path/to/data:/data \
    cooperscene:latest \
    bash
```

The entrypoint runs `python models/bevfusion/setup.py develop` from the project
root if `bev_pool_ext*.so` is not already present, then exec's your command.
Subsequent runs skip the build. The cooperative configs (`configs/cobevt/`,
`configs/v2vam/`, `configs/v2vnet/`, `configs/v2xvit/`) do **not** use these
ops, so the build is only needed for BEVFusion configs.

### Local install (no Docker)

On a host that already has a CUDA toolchain and PyTorch ≥ 2.0 (versions pinned in
`requirements.txt`):

```bash
pip install -U openmim
mim install "mmengine>=0.10" "mmcv>=2.0,<2.2" "mmdet>=3.2,<3.4" "mmdet3d>=1.4,<1.5"
pip install "numpy<1.25" spconv-cu120 einops tqdm pyyaml
# BEVFusion configs additionally need the CUDA ops (run once, from the repo root):
python models/bevfusion/setup.py develop --user
```

Use `spconv-cu120` for CUDA 12.x (or the wheel matching your toolkit). `--user`
is required when site-packages is read-only (Apptainer, locked-down images).
`setup.py` resolves sources relative to the project root, so it must be invoked
from there — not from inside `models/bevfusion/`.

---

## Data Download & Preparation

### Download

*TBD* — public release link will be posted at
<https://data.ucr.edu/datasets/cooperscene/>.

The benchmark ships in **CooperScene format**:
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

> The `mcap/` recordings are distributed separately and are **not** part of the
> core `train/`/`validate/`/`test/` archive — they are only needed for raw replay
> /visualization, not for training or evaluation.

A **mini set** of 180 contiguous frames (120 train / 30 validate / 30 test) 
is shipped alongside the full release for pipeline smoke tests — same `<split>/<take>/<agent>/<frame>` layout, rooted at
`/mini`.

### Data preparation

What you need to prepare depends on the model family.

**Cooperative models** (CoBEVT / V2VAM / V2VNet / V2X-ViT) — **no conversion
needed.** These run through the cooperative runner (OpenCOOD
`IntermediateFusionDataset`), which reads the raw scene folders directly. Point
`data_root` at the directory containing `train/`, `validate/`, `test/` and train
— no `.pkl` index and no `.bin` conversion are required.

**BEVFusion** (single-agent lidar / lidar-cam) — runs on the mmdet3d
`CooperSceneDataset` path and needs `.pkl` info files plus `.bin` point clouds.
Generate them with:

```bash
python tools/dataset_converters/data_converter.py \
    --data-root /path/to/cooperscene \
    --out-dir   /path/to/cooperscene \
    --convert-pcd
```

This writes `cooperscene_infos_{train,val,test}.pkl` into `--out-dir` (set it equal to
`--data-root` so the configs find them) and, with `--convert-pcd`, a `.bin` next
to each `.pcd`. Drop `--convert-pcd` on later runs. `--data-root` must contain
`train/`, `validate/`, `test/`.

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

> **CooperScene ego note:** agent `0` is the LiDAR-only **RSU** and should never
> be the ego. The default `ego_candidates=None` falls back to the smallest
> `cav_id` (agent `0`), which is wrong for CooperScene — always set
> `ego_candidates=['1','2','3']` (the camera-equipped CAVs) for cooperative runs.

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
