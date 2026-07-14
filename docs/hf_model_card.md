---
license: mit
library_name: pytorch
tags:
- 3d-object-detection
- cooperative-perception
- autonomous-driving
- lidar
- camera
- v2x
---

# CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

[![Website](https://img.shields.io/badge/Website-CooperScene-blue?style=for-the-badge)](https://cisl.ucr.edu/CooperScene)
[![Code](https://img.shields.io/badge/Code-CooperScene-181717.svg?style=for-the-badge&logo=github)](https://github.com/UCR-CISL/CooperScene)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/cisl-hf/CooperScene)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/UCR-CISL/CooperScene/blob/main/LICENSE)

## Introduction

🚗 This repository hosts the **model configs and pre-trained checkpoints** for
[CooperScene](https://cisl.ucr.edu/CooperScene) — the first real-world,
multi-agent, multi-modal cooperative autonomy dataset with C-V2X communication
characterization (three connected vehicles + one roadside unit, across
intersections, highway ramps, and parking areas).

🚀 All training and inference code is open-sourced. See the
[project page](https://cisl.ucr.edu/CooperScene) and the
[GitHub repo](https://github.com/UCR-CISL/CooperScene) for details.

💬 We welcome feedback and look forward to your comments!

## What's here

Each model has its config and matching checkpoint together under
`configs/<model>/`:

- **BEVFusion** (single / cooperative × lidar / lidar-cam): `bevfusion_single_lidar`, `bevfusion_single_lidarcam`, `bevfusion_coop_lidar`, `bevfusion_coop_lidarcam`
- **CoBEVT, CoSDH, ERMVP, V2VAM, V2VNet, V2X-ViT**: `cobevt`, `cosdh`, `ermvp`, `v2vam`, `v2vnet`, `v2xvit`

All models run on a unified mmengine pipeline (`proj_first=True`, same global-sort
BEV/3D polygon-IoU AP @ 0.3 / 0.5 / 0.7).

## Download

```bash
pip install -U huggingface_hub
hf download cisl-hf/CooperScene --local-dir assets
# -> assets/configs/<model>/{<model>.py, <model>.pth}
```

## Usage

Clone the [code repo](https://github.com/UCR-CISL/CooperScene), then evaluate or
train with a downloaded config + checkpoint:

```bash
# evaluate (test split by default)
python tools/test.py assets/configs/ermvp/ermvp.py assets/configs/ermvp/ermvp.pth

# train (warm-start from a checkpoint, optional)
python tools/train.py assets/configs/ermvp/ermvp.py
```

See the [GitHub README](https://github.com/UCR-CISL/CooperScene) for data
preparation and the Docker workflow.

## License

The code and the pre-trained checkpoints in this repository are released under
the [MIT License](https://github.com/UCR-CISL/CooperScene/blob/main/LICENSE).
Note that all checkpoints were trained on the CooperScene dataset, which is
licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(see [DATA_LICENSE](https://github.com/UCR-CISL/CooperScene/blob/main/DATA_LICENSE));
the dataset's non-commercial terms apply to any use of the dataset itself.

## Related links

🌐 Website: [https://cisl.ucr.edu/CooperScene](https://cisl.ucr.edu/CooperScene)

💻 GitHub: [https://github.com/UCR-CISL/CooperScene](https://github.com/UCR-CISL/CooperScene)

🤗 Hugging Face: [https://huggingface.co/cisl-hf/CooperScene](https://huggingface.co/cisl-hf/CooperScene)
