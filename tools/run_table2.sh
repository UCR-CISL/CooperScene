#!/usr/bin/env bash
# Reproduce CooperScene Table 2: mAP under C-V2X and Unlimited networks, across
# all agent settings, for each cooperative model. Runs tools/test.py per
# (model, agent_setting, network) and writes metrics under $OUT/<tag>/.
#
# Agent settings (paper Table 2):
#   V+I  V+V  V+V+I  V+2V  V+2V+I   (each averaged over all valid combos)
# Networks:
#   unlimited -> mAP (Unlimited)    cv2x -> mAP (C-V2X)
#
# C-V2X throughput differs PER agent setting (more agents -> more channel
# contention -> lower throughput). Defaults below; override per setting via env
# with the key sanitized (+ -> _), e.g. TP_V_2V_I=0.9 for V+2V+I.
#
# Override via env, e.g.:
#   MODELS="v2vnet cobevt" SETTINGS="V+I V+2V+I" NETWORKS=cv2x bash tools/run_table2.sh
set -euo pipefail

ASSETS=${ASSETS:-assets/configs}          # <ASSETS>/<model>/<model>.{py,pth}
OUT=${OUT:-work_dirs/table2}
MODELS=${MODELS:-"v2vnet v2xvit v2vam cobevt cosdh ermvp"}
SETTINGS=${SETTINGS:-"V+I V+V V+V+I V+2V V+2V+I"}
NETWORKS=${NETWORKS:-"unlimited cv2x"}
BS=${BS:-4}                                # test_dataloader batch size
# Dataset root holding cooperscene_coop_infos_test.pkl (+ the split dirs).
# Default = config's data/cooperscene; override e.g.
#   DATA_ROOT=/workspace/data/Cooperscene/release/250928_opv2v
DATA_ROOT=${DATA_ROOT:-}

# Per-setting C-V2X throughput (Mbps). EDIT these to your measured values.
declare -A TP=( [V+I]=1.6 [V+V]=1.6 [V+V+I]=1.6 [V+2V]=1.6 [V+2V+I]=1.6 )

# Per-cooperator share size (MB) derived from each model's compressed BEV
# feature (tools/calc_sharing.py). Override SHARE_<model> in env if needed.
mkdir -p "$OUT"
for m in $MODELS; do
  cfg="$ASSETS/$m/$m.py"; ckpt="$ASSETS/$m/$m.pth"
  ovr="SHARE_${m}"
  share=${!ovr:-$(python tools/calc_sharing.py --model "$m" --percoop)}
  for net in $NETWORKS; do
    for s in $SETTINGS; do
      tpkey="TP_${s//+/_}"                 # V+2V+I -> TP_V_2V_I
      tp=${!tpkey:-${TP[$s]}}
      tag="${m}__${s//+/-}__${net}"
      echo "=== $tag (share=${share}MB @ ${tp}Mbps) ==="
      opts=(
        test_dataloader.batch_size="$BS"
        test_dataloader.dataset.agent_setting="$s"
        test_dataloader.dataset.network="$net"
        test_dataloader.dataset.share_size_mb="$share"
        test_dataloader.dataset.cv2x_throughput="$tp"
      )
      [ -n "$DATA_ROOT" ] && opts+=( test_dataloader.dataset.data_root="$DATA_ROOT" )
      python tools/test.py "$cfg" "$ckpt" \
        --work-dir "$OUT/$tag" \
        --cfg-options "${opts[@]}" \
        2>&1 | tee "$OUT/$tag.log"
    done
  done
done
echo "Done. Metrics under $OUT/<model>__<setting>__<network>/"
