#!/usr/bin/env bash
# Precompute the demo's preset outputs on the test split: per ego sample the
# predicted boxes, GT and the fused cooperators (+ delays) for each
# (model, agent setting, network). One pickle per cell (tools/test_dump.py).
#
#   V        -> ego only ("single-vehicle detections"), network-independent
#   V+I, V+2V, V+2V+I -> one cooperator combination per ego frame (V+V and
#                        V+V+I have two, so they are not used for the viewer);
#                        ego frames whose peers are out of range are skipped
#   all      -> agent_setting=None: every in-range cooperator, so EVERY ego
#               frame has a record (the set the viewer should default to)
#   networks -> unlimited | cv2x (fixed 1.6 Mbps) | trace (per-frame replay)
#
# Share sizes are the PUBLISHED Table 1 per-cooperator sizes (measured sparse
# sizes for CoSDH / ERMVP), not the analytic dense sizes of calc_sharing.py.
#
#   MODELS="cosdh" DATA_ROOT=/workspace/data/Cooperscene/release/250928_opv2v bash tools/run_demo_dumps.sh
set -euo pipefail

MODELS=${MODELS:-"cosdh cobevt"}
SETTINGS=${SETTINGS:-"V V+I V+2V V+2V+I all"}
NETWORKS=${NETWORKS:-"unlimited cv2x trace"}
TRACE_MAX_DELAY=${TRACE_MAX_DELAY:-30}
OUT=${OUT:-work_dirs/dumps}
DATA_ROOT=${DATA_ROOT:-}
BS=${BS:-4}
declare -A SHARE=( [cosdh]=0.0595 [ermvp]=0.346 [cobevt]=0.500 [v2xvit]=0.338 [v2vam]=0.423 [v2vnet]=10.0 )

mkdir -p "$OUT"
for m in $MODELS; do
  cfg="assets/configs/$m/$m.py"; ckpt="assets/configs/$m/$m.pth"
  share=${SHARE[$m]}
  for s in $SETTINGS; do
    nets="$NETWORKS"; [ "$s" = "V" ] && nets="unlimited"
    for net in $nets; do
      tag="${m}__${s//+/-}__${net}"
      [ "$net" = "trace" ] && tag="${tag}${TRACE_MAX_DELAY}"
      if [ -s "$OUT/$tag.pkl" ]; then echo "skip $tag (exists)"; continue; fi
      echo "=== $tag (share=${share}MB) ==="
      setting="$s"; [ "$s" = "all" ] && setting="None"
      opts=(
        test_dataloader.batch_size="$BS"
        test_dataloader.dataset.agent_setting="$setting"
        test_dataloader.dataset.network="$net"
        test_dataloader.dataset.share_size_mb="$share"
        test_dataloader.dataset.trace_max_delay="$TRACE_MAX_DELAY"
      )
      [ -n "$DATA_ROOT" ] && opts+=( test_dataloader.dataset.data_root="$DATA_ROOT" )
      python tools/test_dump.py "$cfg" "$ckpt" \
        --out "$OUT/$tag.pkl" --work-dir "$OUT/$tag" \
        --cfg-options "${opts[@]}" 2>&1 | tee "$OUT/$tag.log"
    done
  done
done
echo "Done. Dumps under $OUT/<model>__<setting>__<network>.pkl"
