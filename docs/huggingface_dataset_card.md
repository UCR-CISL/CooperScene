---
# Hugging Face dataset card for CooperScene.
# Upload this file as the README.md of the Hugging Face dataset repo so that
# anyone who lands on Hugging Face first is pointed straight back to the code
# repository and the official data release.
pretty_name: CooperScene
license: cc-by-nc-sa-4.0          # TODO: confirm the actual release license
language:
  - en
task_categories:
  - object-detection
tags:
  - autonomous-driving
  - cooperative-perception
  - v2x
  - c-v2x
  - lidar
  - multi-modal
  - 3d-object-detection
size_categories:
  - 10K<n<100K
---

# CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

**Code, configs, training & evaluation:** https://github.com/UCR-CISL/CooperScene
**Dataset on Hugging Face:** https://huggingface.co/cisl-hf/CooperScene
**Official data release:** https://data.ucr.edu/datasets/cooperscene/
**Paper:** TBD &nbsp;|&nbsp; **Project website:** TBD

> This Hugging Face repo hosts the **dataset** only. To reproduce the benchmark —
> dataset converters, model configs (CoBEVT / V2VAM / V2VNet / V2X-ViT /
> BEVFusion), the Docker image, and training/evaluation scripts — use the code
> repository: **https://github.com/UCR-CISL/CooperScene**.

## Overview

CooperScene is the first real-world, multi-agent, multi-modal cooperative
autonomy dataset with C-V2X communication characterization. It features three
connected autonomous vehicles (CAVs) and one instrumented infrastructure roadside
unit (RSU), all equipped with multi-modal sensors and commercial off-the-shelf
C-V2X radios, interacting across diverse real-world traffic scenarios including
intersections, highway ramps, and parking areas.

- 59K synchronized LiDAR frames across 4 cooperative agents
- 53K camera image frames
- 344K globally consistent 3D bounding box labels at 10 Hz
- Real-world C-V2X communication traces (latency, throughput, packet loss, jitter)
- Centimeter-level localization via GNSS-RTK + spatial-temporal ICP alignment
- Sub-millisecond sensor synchronization via PTP and hardware triggering

## Data layout

The benchmark ships in CooperScene format:
`<split>/<take>/<agent>/<frame>.{pcd,yaml}` plus `<frame>_camera0.png` on
camera-equipped agents. Each take has 4 agents: agent `0` is LiDAR-only (the RSU);
agents `1-3` also carry a front camera.

```
cooperscene/
├── train/
│   └── <take>/<agent>/<frame>.{pcd,yaml}[, <frame>_camera0.png]
├── validate/
└── test/
```

A mini set of 180 contiguous frames (120 train / 30 validate / 30 test) is
provided for quick pipeline smoke tests.

## Getting started

Clone the code repo and follow its README for environment setup (Docker or local),
data preparation, and training/evaluation:

```bash
git clone https://github.com/UCR-CISL/CooperScene.git
cd CooperScene
# see README.md for Docker build, data preparation, and training commands
```

## License

TODO: confirm and state the official license for the dataset before publishing.

## Citation

```bibtex
@misc{cooperscene,
  title  = {CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization},
  author = {UCR-CISL},
  year   = {2025},
  note   = {https://github.com/UCR-CISL/CooperScene}
}
```
<!-- TODO: replace with the final BibTeX entry once the paper is public. -->
