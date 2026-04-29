# CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

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

## Repository Structure

```
CooperScene/
├── projects/
│   ├── coop/                          # Cooperative 3D detection (CoBEVT, V2VAM, V2VNet, V2X-ViT)
│   │   ├── configs/opv2v/             # Training configs
│   │   └── mmdet3d_plugin/            # Datasets, models, metric, preprocessor
│   └── BEVFusion/                     # Multi-modal cooperative perception (LiDAR / LiDAR+Camera)
│       ├── configs/                   # BEVFusion configs (single-agent, coop, CooperScene)
│       └── bevfusion/                 # Plugin code + CUDA ops
├── tools/
│   ├── train.py / test.py             # Entry points (mmengine runner)
│   └── dataset_converters/            # OPV2V → mmdet3d-pkl converters
├── OpenCOOD-modified/                 # Original OpenCOOD pipeline (separate baseline)
├── docs/                              # Long-form documentation
│   ├── data_preparation.md
│   └── training.md
├── requirements.txt
└── README.md
```

`projects/` follows the **mmdet3d plugin** convention (à la UniAD / FusionAD):
no fork of `mmdet3d`, custom code is loaded by each config's `custom_imports`.

---

## Quick Start

### Docker (recommended)

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
cd projects/BEVFusion && python setup.py develop && cd ../..
```

This compiles `bev_pool_ext` and the voxel ops needed by the BEVFusion configs.
Cooperative configs under `projects/coop/` do **not** require this step.

---

## Data Downloading

*TBD* — public release link will be posted here. The benchmark currently consumes
OPV2V-format raw scenes (`.pcd` LiDAR + `.png` images + per-frame `.yaml` poses);
CooperScene shares the same format.

---

## Data Preparation

mmdet3d-style `.pkl` info files are generated from raw OPV2V/CooperScene scenes
by the converters under `tools/dataset_converters/`.

See **[docs/data_preparation.md](docs/data_preparation.md)** for the full
walkthrough (expected directory layout, single-agent vs. cooperative info,
flag reference, output schema).

---

## Train Your Model

Cooperative 3D detection (CoBEVT / V2VAM / V2VNet / V2X-ViT) and BEVFusion
multi-modal training/testing instructions live in
**[docs/training.md](docs/training.md)**.

The doc covers:
- Training each of the four cooperative models on OPV2V
- BEVFusion training pipeline (single-agent → coop, LiDAR → LiDAR+Camera, OPV2V → CooperScene transfer)
- Single-GPU / multi-GPU launches
- Testing & metric reporting (`AP_BEV@{0.3,0.5,0.7}`, `AP_3D@{0.3,0.5,0.7}`)
- Common overrides: AMP, batch size, resume

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
