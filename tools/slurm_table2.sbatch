#!/usr/bin/env bash
# Run tools/run_table2.sh inside the OpenCOOD/mmdet3d container on the eegrad
# cluster. Every knob of run_table2.sh is passed through the environment:
#
#   sbatch --export=ALL,MODELS="cosdh ermvp",NETWORKS=trace,TRACE_MAX_DELAY=30 \
#          tools/slurm_table2.sbatch
#
# Partition notes (2026-10): dune2 is Blackwell (container torch unsupported),
# node04 fails at start, node05 silently returns empty evals -> all excluded.
#SBATCH --job-name=cs_table2
#SBATCH --partition=cisl,batch
#SBATCH --exclude=dune2,node04,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/eegrad/bowu/CooperScene/work_dirs/slurm/%x_%j.out
set -euo pipefail

REPO=/home/eegrad/bowu/CooperScene
SIF=/home/eegrad/bowu/OpenCOOD-modified/opencood.sif
mkdir -p "$REPO/work_dirs/slurm"

echo "host $(hostname) job $SLURM_JOB_ID gpu_bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1)"
echo "MODELS=${MODELS:-} SETTINGS=${SETTINGS:-} NETWORKS=${NETWORKS:-} TRACE_MAX_DELAY=${TRACE_MAX_DELAY:-} TRACE_DROP=${TRACE_DROP:-} OUT=${OUT:-}"

# Configs use data_root='data/cooperscene'; the released benchmark lives at
# /workspace/data/Cooperscene/release/250928_opv2v inside the container and
# the pkls point at the *_bin copy for point clouds (both under /data/HangQiu).
apptainer exec --nv \
  --bind "$REPO":/workspace \
  --bind /data/HangQiu/data:/workspace/data \
  --pwd /workspace \
  --env PYTHONPATH=/workspace \
  --env DATA_ROOT=/workspace/data/Cooperscene/release/250928_opv2v \
  --env MODELS="${MODELS:-v2vnet v2xvit v2vam cobevt cosdh ermvp}" \
  --env SETTINGS="${SETTINGS:-V+I V+V V+V+I V+2V V+2V+I}" \
  --env NETWORKS="${NETWORKS:-unlimited cv2x}" \
  --env TRACE_MAX_DELAY="${TRACE_MAX_DELAY:-30}" \
  --env TRACE_DROP="${TRACE_DROP:-True}" \
  --env OUT="${OUT:-work_dirs/table2}" \
  --env BS="${BS:-4}" \
  "$SIF" bash tools/run_table2.sh
