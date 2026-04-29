# Data Preparation

mmdet3d-style training/eval requires per-sample `.pkl` info files alongside the
raw scenes. CooperScene ships two converters — one for single-agent evaluation
(OPV2V-style flat scenes) and one for cooperative evaluation (groups all CAVs at
the same timestamp into one sample with N "ego rotations"):

| Converter | Output | Used by |
|---|---|---|
| `tools/dataset_converters/opv2v_converter.py` | `opv2v_infos_{train,val,test}.pkl` | Single-agent BEVFusion configs |
| `tools/dataset_converters/opv2v_cooper_converter.py` | `opv2v_coop_infos_{train,val,test}.pkl` | Cooperative configs (CoBEVT / V2VAM / V2VNet / V2X-ViT, coop BEVFusion) |

The cooperative converter internally calls helpers from `opv2v_converter.py`
(camera intrinsics parsing, `.pcd` → `.bin` conversion), so keep both files in
the same directory.

---

## Expected Raw Layout

OPV2V and CooperScene share the same directory convention. Place the raw scenes
anywhere (e.g. `/data/OPV2V`):

```
/data/OPV2V/
├── train/
│   └── <scenario>/
│       └── <agent_id>/
│           ├── <timestamp>.pcd          # LiDAR (or .bin if pre-converted)
│           ├── <timestamp>.yaml         # Pose, ego transform, GT bboxes
│           └── camera0..3/
│               └── <timestamp>.png      # 4 camera views
├── validate/
└── test/
```

`<scenario>` and `<agent_id>` are arbitrary string IDs; the converter discovers
them by directory listing.

---

## Generate Info Files

### Cooperative (CoBEVT / V2VAM / V2VNet / V2X-ViT, coop BEVFusion)

```bash
python tools/dataset_converters/opv2v_cooper_converter.py \
    --data-root /data/OPV2V \
    --out-dir   /data/OPV2V \
    --splits train validate test \
    --convert-pcd \
    --num-workers 8
```

Flags:
- `--data-root` — directory containing `train/ validate/ test/` scenes.
- `--out-dir` — where the `.pkl` files are written. **Must match
  `data_root` in the configs** (configs default to `/data/OPV2V`).
- `--splits` — which splits to process; converter maps `validate → val`
  internally, so the produced filenames are
  `opv2v_coop_infos_train.pkl`, `opv2v_coop_infos_val.pkl`,
  `opv2v_coop_infos_test.pkl`.
- `--convert-pcd` — converts `<timestamp>.pcd` → `<timestamp>.bin`. **Pass
  this on the very first run only**, then drop it on subsequent runs.
- `--num-workers` — parallel scenario processing.

### Single-Agent (BEVFusion `bevfusion_lidar_*` / `bevfusion_lidar-cam_*`)

```bash
python tools/dataset_converters/opv2v_converter.py \
    --data-root /data/OPV2V \
    --out-dir   /data/OPV2V \
    --splits train validate test \
    --convert-pcd
```

Produces `opv2v_infos_{train,val,test}.pkl`. These are needed only by the
single-agent BEVFusion configs (Step 1 / Step 2 of the training pipeline in
[docs/training.md](training.md)).

### CooperScene

CooperScene uses the same directory convention as OPV2V; point `--data-root`
at your CooperScene root. The cooperative converter produces
`opv2v_coop_infos_*.pkl` (filename retained for config compatibility — feel
free to symlink or rename). For BEVFusion's CooperScene configs, look at the
specific config to confirm which `.pkl` filename it expects.

---

## Output `.pkl` Schema (cooperative)

Each entry in the cooperative `.pkl` describes one **(scenario, timestamp)
group**, expanded into N samples (one per CAV taking its turn as "ego"):

```python
{
    'sample_idx': str,                     # "<scenario>_<timestamp>_<ego_id>"
    'lidar_path': str,                     # ego's .bin path
    'images': {camera0..3: {...}},         # ego's 4 cameras with intrinsics
    'gt_bboxes_3d': np.ndarray (N, 7),     # GT in ego frame
    'gt_labels_3d': np.ndarray (N,),       # 0 = vehicle
    'agents': [                            # all CAVs in this scene/timestamp
        {
            'agent_id': str,
            'is_ego': bool,
            'lidar_path': str,             # this agent's .bin
            'images': {camera0..3: {...}},
            'lidar_pose': np.ndarray (4, 4),
            'agent2ego': np.ndarray (4, 4),
        },
        ...
    ],
}
```

The `agents` list is what `LoadCooperativePointCloud` consumes to build the
ego-frame multi-CAV point cloud and the `(max_cav, max_cav, 4, 4)`
`pairwise_t_matrix`.

---

## Common Issues

- **`FileNotFoundError: ..._cam.png`** — converter expects `camera0..3/`
  subdirs; if your raw scenes don't have them, the per-camera entries will be
  silently skipped.
- **Slow first run** — `--convert-pcd` decodes every `.pcd`; subsequent runs
  without that flag take seconds.
- **Mismatch with `data_root` in configs** — if you write the `.pkl` somewhere
  other than the path in the config (`data_root='/data/OPV2V'` by default),
  override with `--cfg-options data_root=<your_path>` at training time, or edit
  the config.
