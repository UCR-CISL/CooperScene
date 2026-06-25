# CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

[**arXiv**](TBD) &nbsp;|&nbsp; [**Project Website**](https://cisl.ucr.edu/CooperScene) &nbsp;|&nbsp; [**Hugging Face**](https://huggingface.co/cisl-hf/CooperScene)

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

## Data Download & Preparation

### Download

Public release link
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
is shipped alongside the full release for pipeline smoke tests — same `<split>/<take>/<agent>/<frame>` layout.

### Data preparation

All models run on the mmengine pipeline and need `.pkl` index files plus `.bin`
point clouds. Run the converter **once** with `--convert-pcd` (writes a `.bin`
next to every `.pcd`); drop the flag on later runs. `--data-root` must contain
`train/`, `validate/`, `test/`.


```bash
python tools/dataset_converters/coop_data_converter.py \
    --data-root /path/to/cooperscene \
    --convert-pcd
# -> <data-root>/cooperscene_coop_infos_{train,val,test}.pkl
```

**Single-agent BEVFusion**:

```bash
python tools/dataset_converters/data_converter.py \
    --data-root /path/to/cooperscene \
    --out-dir   /path/to/cooperscene \
    --convert-pcd
# -> <out-dir>/cooperscene_infos_{train,val,test}.pkl
```

---

## Quick Start

### Docker

The image builds on `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel` and installs
the full stack (mmengine / mmcv 2.1 / mmdet 3.x / mmdet3d 1.4 / spconv /
shapely). 

#### 1. Build the image

```bash
cd CooperScene/
docker build -t cooperscene -f docker/Dockerfile .
```

#### 2. Get configs + checkpoints from Hugging Face

```bash
cd CooperScene/
pip install -U huggingface_hub
hf download cisl-hf/CooperScene --local-dir assets
# -> assets/configs/<model>/{<model>.py, <model>.pth}  (config + checkpoint together)
```

#### 3. Enter the container (bind code + dataset)

All configs use `data_root = 'data/cooperscene'`, so bind your dataset to
`…/data/cooperscene` and the commands below need no path overrides.

```bash
cd CooperScene/
docker run --gpus all -it --rm \
    -v "$(pwd)":/workspace/CooperScene \
    -v /path/to/cooperscene_dataset:/workspace/CooperScene/data/cooperscene \
    cooperscene bash
```

#### 4. Train (inside the container)
`tools/train.py` trains models on **train** split, and validates on **validate** split
```bash
python tools/train.py assets/configs/ermvp/ermvp.py
```

Available configs (swap the path above for any of these):

- **BEVFusion** (single / cooperative × lidar / lidar-cam):
  `bevfusion/bevfusion_single_lidar.py`, `bevfusion/bevfusion_single_lidarcam.py`,
  `bevfusion/bevfusion_coop_lidar.py`, `bevfusion/bevfusion_coop_lidarcam.py`
- **CoBEVT / CoSDH / ERMVP / V2VAM / V2VNet / V2X-ViT**:
  `cobevt/cobevt.py`, `cosdh/cosdh.py`, `ermvp/ermvp.py`,
  `v2vam/v2vam.py`, `v2vnet/v2vnet.py`, `v2xvit/v2xvit.py`

#### 5. Evaluate

`tools/test.py` evaluates the **test** split by default. Each config has a
matching checkpoint in the same folder (`<model>.pth`):

```bash
# ERMVP
python tools/test.py assets/configs/ermvp/ermvp.py assets/configs/ermvp/ermvp.pth

# BEVFusion (cooperative lidar)
python tools/test.py assets/configs/bevfusion/bevfusion_coop_lidar.py \
                     assets/configs/bevfusion/bevfusion_coop_lidar.pth
```

#### Table 2: agent settings x network

The cooperative `CoopDataset` accepts two eval-time knobs (override with
`--cfg-options test_dataloader.dataset.<field>=...`):

| Field | Values | Meaning |
|---|---|---|
| `agent_setting` | `V+I`, `V+V`, `V+V+I`, `V+2V`, `V+2V+I`, `None` | which cooperators participate; sub-settings are averaged over every valid agent combination. `None` = full V+2V+I, no expansion |
| `network` | `unlimited`, `cv2x` | `unlimited` = perfect sharing (mAP Unlimited); `cv2x` = async transmission delay from `share_size_mb` (per cooperator, Table 2) and `cv2x_throughput` (default 1.6 Mbps), giving mAP (C-V2X) |

Single setting example (V+V over C-V2X for v2vnet):

```bash
python tools/test.py assets/configs/v2vnet/v2vnet.py assets/configs/v2vnet/v2vnet.pth \
    --cfg-options \
        test_dataloader.dataset.agent_setting=V+V \
        test_dataloader.dataset.network=cv2x \
        test_dataloader.dataset.share_size_mb=10.0
```

Full Table 2 sweep (all models x settings x networks):

```bash
bash tools/run_table2.sh                 # -> work_dirs/table2/<model>__<setting>__<network>/
MODELS="cobevt" SETTINGS="V+I V+2V+I" NETWORKS=cv2x bash tools/run_table2.sh
```


### Local install

Tested with Python 3.10 / CUDA 12.1 / PyTorch 2.1. Dependencies are listed in
`requirements.txt` (use `mim` so the correct `mmcv` wheel is fetched):

```bash
cd CooperScene/
pip install -U openmim
mim install -r requirements.txt
# BEVFusion configs need the CUDA ops, built once from the repo root:
python models/bevfusion/setup.py develop
```

---

## Arguments


Common parameters you'll override most often:

| Field | Meaning |
|---|---|
| `train_dataloader.batch_size` | per-GPU batch (default 4) |
| `train_dataloader.num_workers` | worker processes (default 4) |
| `*_dataloader.dataset.data_root` | dataset root containing `train/`, `validate/`, `test/` (default `data/cooperscene`) |
| `*_dataloader.dataset.ann_file` | `.pkl` index for that split (e.g. `cooperscene_coop_infos_val.pkl`) |
| `optim_wrapper.optimizer.lr` | base learning rate (per-config) |
| `train_cfg.max_epochs` | total epochs (per-config) |
| `load_from` | warm-start checkpoint path (default `None` = train from scratch) |

Override any of these from the CLI with `--cfg-options`. Example (ermvp):

```bash
python tools/train.py assets/configs/ermvp/ermvp.py \
    --cfg-options \
        train_dataloader.batch_size=2 \
        train_dataloader.num_workers=8 \
        train_dataloader.dataset.data_root=/path/to/cooperscene \
        train_cfg.max_epochs=30 \
        optim_wrapper.optimizer.lr=5e-4 \
        load_from=/path/to/ckpt.pth
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
