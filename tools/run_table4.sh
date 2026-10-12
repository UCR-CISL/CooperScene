#!/usr/bin/env bash
# Table 4: multi-modal cooperative BEVFusion (LiDAR / LiDAR+Cam) across agent
# settings V / V+V / V+2V (vehicles only, no infra). V uses the single-agent
# model; V+V and V+2V use the cooperative model with agent_setting.
#
#   V     -> bevfusion_single_<mod> on cooperscene_infos_test.pkl (3 vehicle egos)
#   V+V   -> bevfusion_coop_<mod>,  agent_setting=V+V  (1 peer vehicle)
#   V+2V  -> bevfusion_coop_<mod>,  agent_setting=V+2V (2 peer vehicles)
#
# Assets (config + ckpt) under $ASSETS; get them with:
#   hf download cisl-hf/CooperScene --include "configs/bevfusion/*" --local-dir assets
set -euo pipefail

ASSETS=${ASSETS:-assets/configs/bevfusion}
OUT=${OUT:-work_dirs/table4}
MODALITIES=${MODALITIES:-"lidar lidarcam"}
SETTINGS=${SETTINGS:-"V V+V V+2V"}
DATA_ROOT=${DATA_ROOT:-}
BS=${BS:-4}                                # test batch size (config default is 1)

mkdir -p "$OUT"
for mod in $MODALITIES; do
  for s in $SETTINGS; do
    if [ "$s" = "V" ]; then
      cfg="$ASSETS/bevfusion_single_${mod}.py"; ckpt="$ASSETS/bevfusion_single_${mod}.pth"
      opts=( test_dataloader.dataset.ann_file=cooperscene_infos_test.pkl )
    else
      cfg="$ASSETS/bevfusion_coop_${mod}.py";   ckpt="$ASSETS/bevfusion_coop_${mod}.pth"
      opts=( test_dataloader.dataset.agent_setting="$s" )
    fi
    opts+=( test_dataloader.batch_size="$BS" )
    [ -n "$DATA_ROOT" ] && opts+=( test_dataloader.dataset.data_root="$DATA_ROOT" )
    tag="bevfusion_${mod}__${s//+/-}"
    echo "=== $tag ==="
    python tools/test.py "$cfg" "$ckpt" \
      --work-dir "$OUT/$tag" \
      --cfg-options "${opts[@]}" \
      2>&1 | tee "$OUT/$tag.log"
  done
done
echo "Done. Metrics under $OUT/bevfusion_<modality>__<setting>/"
