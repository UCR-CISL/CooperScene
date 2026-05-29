#!/usr/bin/env bash
set -e

# Build the BEVFusion CUDA ops (bev_pool_ext) on first container start.
# Skips if the .so is already present (e.g. user pre-built outside the
# container, or this is a subsequent run on the same mounted source tree).
BEV_OPS_DIR="${COOPERSCENE_ROOT:-/workspace/CooperScene}/models/bevfusion"
if [ -d "$BEV_OPS_DIR" ]; then
    if ! ls "$BEV_OPS_DIR"/ops/bev_pool/bev_pool_ext*.so >/dev/null 2>&1; then
        echo "[entrypoint] Building BEVFusion CUDA ops in $BEV_OPS_DIR ..."
        # `--user` keeps the egg-info install path writable when site-packages
        # is read-only (Apptainer, locked-down Docker images, etc.). The .so
        # itself lands inside the source tree either way.
        (cd "$BEV_OPS_DIR" && python setup.py develop --user)
        echo "[entrypoint] Build done."
    fi
fi

exec "$@"
