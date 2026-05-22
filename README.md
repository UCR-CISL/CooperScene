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

A pre-built image with all Python + CUDA dependencies (PyTorch 2.1.1, mmdet3d 1.4,
mmcv 2.1, spconv 2.x, etc.):

```bash
docker pull bwu109/motion_prediction@sha256:32e06e6533ce82d267696b8821b9f494d2f508971ab5501e736a65f1fb1ddcc3

docker run --gpus all -it --rm \
    -v $(pwd):/workspace/CooperScene \
    -v /path/to/data:/data \
    bwu109/motion_prediction@sha256:32e06e6533ce82d267696b8821b9f494d2f508971ab5501e736a65f1fb1ddcc3 \
    bash
```

(Other tags: <https://hub.docker.com/repository/docker/bwu109/motion_prediction/tags>.)

### Build BEVFusion CUDA ops (one-time)

```bash
cd models/bevfusion && python setup.py develop && cd ../..
```

This compiles `bev_pool_ext` and the voxel ops needed by the BEVFusion configs
under `configs/bevfusion/`. The other cooperative configs (`configs/cobevt/`,
`configs/v2vam/`, `configs/v2vnet/`, `configs/v2xvit/`) do **not** require
this step.

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
└── test/
```

A **mini set** of 180 contiguous frames (120 train / 30 validate / 30 test,
all from one take, 4 agents aligned) is shipped alongside the full release for
pipeline smoke tests — same `<split>/<take>/<agent>/<frame>` layout, rooted at
`cooperscene_mini/`.

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
├── bevfusion/   # BEVFusion (single-agent + cooperative)
├── cobevt/      # PointPillars + CoBEVT
├── v2vam/       # PointPillars + V2VAM
├── v2vnet/      # PointPillars + V2VNet
└── v2xvit/      # PointPillars + V2X-ViT
```

Each method's `_base_` (runtime defaults) is `configs/_base_/default_runtime.py`.

### Train

```bash
python tools/train.py configs/cobevt/pointpillars_cobevt.py
```

Swap in any other config under `configs/`. Checkpoints and logs land under
`work_dirs/<config_stem>/`.

### Inference

```bash
python tools/test.py configs/cobevt/pointpillars_cobevt.py \
    work_dirs/pointpillars_cobevt/epoch_90.pth
```

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
