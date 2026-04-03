# CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

CooperScene is the first real-world, multi-agent, multi-modal cooperative autonomy dataset with C-V2X communication characterization. It features three connected autonomous vehicles (CAVs) and one instrumented infrastructure roadside unit (RSU), all equipped with multi-modal sensors and commercial off-the-shelf C-V2X communication radios, interacting across diverse real-world traffic scenarios including intersections, highway ramps, and parking areas.

**Key highlights:**
- **59K** synchronized LiDAR frames across 4 cooperative agents
- **53K** camera image frames
- **344K** globally consistent 3D bounding box labels at 10 Hz
- Real-world C-V2X communication traces (latency, throughput, packet loss, jitter)
- Centimeter-level localization via GNSS-RTK + spatial-temporal ICP alignment
- Sub-millisecond sensor synchronization via PTP and hardware triggering

## Repository Structure

This branch contains the benchmark code for CooperScene, organized as two Git submodules:

```
CooperScene/
  OpenCOOD-modified/   # Cooperative 3D object detection (V2VNet, V2X-ViT, V2VAM, CoBEVT)
  mmdetection3d/       # Multi-modal detection (BEVFusion) & motion prediction (CMP)
```

- **[OpenCOOD-modified](https://github.com/UCR-CISL/OpenCOOD-modified)**: Modified [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) framework for cooperative 3D object detection with intermediate fusion methods.
- **[mmdetection3d](https://github.com/UCR-CISL/mmdetection3d-opv2v)**: Fork of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) extended with BEVFusion cooperative perception and end-to-end motion prediction.

## Getting Started

### Clone

```bash
git clone --recurse-submodules -b benchmark https://github.com/UCR-CISL/CooperScene.git
cd CooperScene
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### Data Preparation

Download the dataset from [data.ucr.edu](https://data.ucr.edu) and organize it as follows:

**For OpenCOOD (OPV2V format):**
```
OpenCOOD-modified/
  data/
    opv2v_data_dumping/
      train/
        {scene_id}/{timestamp}/{agent_id}.pcd
        {scene_id}/{timestamp}/{agent_id}.yaml
      validate/
      test/
```

**For mmdetection3d:**
```
mmdetection3d/
  data/
    cooperscene/
      train/
        {scene_id}/{agent_id}/{timestamp}.pcd
        {scene_id}/{agent_id}/{timestamp}.yaml
      validate/
      test/
      cooperscene_infos_train.pkl
      cooperscene_infos_val.pkl
```

Generate info files for mmdetection3d:
```bash
cd mmdetection3d
python tools/dataset_converters/cooperscene_converter.py
```

---

## Benchmark 1: Cooperative 3D Object Detection (OpenCOOD)

Evaluates intermediate fusion methods under both ideal (unlimited) and real C-V2X network conditions across five agent configurations: V+I, V+V, V+V+I, V+2V, V+2V+I.

### Installation

```bash
cd OpenCOOD-modified
conda env create -f environment.yml
conda activate opencood
python setup.py develop
```

Install [spconv](https://github.com/traveller59/spconv) (e.g., for CUDA 11.3):
```bash
pip install spconv-cu113
```

### Training

```bash
# Single GPU
python opencood/tools/train.py --hypes_yaml <CONFIG_FILE>

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 --use_env opencood/tools/train.py \
  --hypes_yaml <CONFIG_FILE>
```

Available configs in `opencood/hypes_yaml/`:
| Model | Config |
|-------|--------|
| V2VNet | `point_pillar_v2vnet.yaml` |
| V2X-ViT | `point_pillar_transformer.yaml` |
| V2VAM | `point_pillar_intermediate_V2VAM.yaml` |
| CoBEVT | `point_pillar_cobevt.yaml` |

### Evaluation

```bash
python opencood/tools/inference.py \
  --model_dir <CHECKPOINT_DIR> \
  --fusion_method intermediate
```

Reports AP@0.3, AP@0.5, AP@0.7 for both BEV and 3D evaluation.

---

## Benchmark 2: Multi-Modal Cooperative Perception (BEVFusion)

Evaluates the impact of multi-modal sensing (LiDAR vs. LiDAR+Camera) under cooperative settings using BEVFusion.

### Installation

```bash
cd mmdetection3d
pip install -r requirements/runtime.txt
pip install -r requirements/mminstall.txt
pip install -v -e .

# Build BEVFusion CUDA ops
python projects/BEVFusion/setup.py develop
```

### Training

```bash
# Single-agent LiDAR
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_cooperscene.py

# Cooperative LiDAR
python tools/train.py projects/BEVFusion/configs/bevfusion_coop_lidar_opv2v.py \
  --cfg-options load_from=<SINGLE_AGENT_CHECKPOINT>

# Cooperative LiDAR+Camera
python tools/train.py projects/BEVFusion/configs/bevfusion_coop_lidarcam_opv2v.py \
  --cfg-options load_from=<LIDAR_CHECKPOINT>

# Multi-GPU
bash tools/dist_train.sh <CONFIG> <NUM_GPUS>
```

### Evaluation

```bash
python tools/test.py <CONFIG> <CHECKPOINT>
```

Reports mAP@0.3, mAP@0.5, mAP@0.7 for both BEV and 3D.

---

## Benchmark 3: Cooperative Motion Prediction

Evaluates cooperative motion prediction using V2VNet and CMP following the perception-prediction (P&P) setting.

### Training

```bash
# End-to-end motion prediction
python tools/train.py projects/UniAD/configs/bevfusion_uniad_e2e_opv2v.py
```

### Evaluation

```bash
python tools/test.py projects/UniAD/configs/bevfusion_uniad_e2e_opv2v.py <CHECKPOINT>
```

Reports minADE and minFDE at horizons of 1s, 3s, and 5s.

---

## Citation

TBD

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) for the cooperative perception framework
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for the 3D detection toolbox
- [BEVFusion](https://github.com/mit-han-lab/bevfusion) for multi-modal BEV fusion
- [UniAD](https://github.com/OpenDriveLab/UniAD) for the end-to-end motion prediction architecture
