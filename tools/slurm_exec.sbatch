#!/usr/bin/env bash
# Run an arbitrary command inside the OpenCOOD/mmdet3d container on one GPU,
# from the repo root (/workspace), with the released dataset bound.
#
#   sbatch tools/slurm_exec.sbatch python tools/test_dump.py <cfg> <ckpt> --out ... --cfg-options ...
#
#SBATCH --job-name=cs_exec
#SBATCH --partition=cisl,batch
#SBATCH --exclude=dune2,node04,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/eegrad/bowu/CooperScene/work_dirs/slurm/%x_%j.out
set -euo pipefail

REPO=/home/eegrad/bowu/CooperScene
SIF=/home/eegrad/bowu/OpenCOOD-modified/opencood.sif
mkdir -p "$REPO/work_dirs/slurm"
echo "host $(hostname) job $SLURM_JOB_ID gpu_bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1)"
echo "cmd: $*"

# The image ships PYTHONPATH=/workspace/opv2v, which hides this repo's
# `models` package; override it so custom_imports resolve.
apptainer exec --nv \
  --bind "$REPO":/workspace \
  --bind /data/HangQiu/data:/workspace/data \
  --pwd /workspace \
  --env PYTHONPATH=/workspace \
  "$SIF" "$@"
