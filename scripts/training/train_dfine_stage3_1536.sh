#!/usr/bin/env bash
set -euo pipefail

: "${DFINE_ROOT:?Set DFINE_ROOT to the upstream D-FINE repository}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${DFINE_STAGE2_CKPT:?Set DFINE_STAGE2_CKPT to the Stage-2 epoch-42 checkpoint}"

cd "$DFINE_ROOT"
python train.py \
  -c "$PROJECT_ROOT/configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1536_v1.yml" \
  --use-amp \
  --seed=0 \
  -t "$DFINE_STAGE2_CKPT"
