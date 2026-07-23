#!/usr/bin/env bash
set -euo pipefail

: "${DFINE_ROOT:?Set DFINE_ROOT to the upstream D-FINE repository}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${DFINE_STAGE1_CKPT:?Set DFINE_STAGE1_CKPT to Stage-1 checkpoint0047.pth}"

cd "$DFINE_ROOT"
python train.py \
  -c "$PROJECT_ROOT/configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1440_v1.yml" \
  --use-amp \
  --seed=0 \
  -t "$DFINE_STAGE1_CKPT"
