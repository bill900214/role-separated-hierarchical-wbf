#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${EVAL_IMAGES:?Set EVAL_IMAGES to FishEye1K_eval images}"

: "${DEIMV2_ROOT:?Set DEIMV2_ROOT to the upstream DEIMv2 repository}"
: "${DEIMV2_832_WEIGHT:?Set DEIMV2_832_WEIGHT}"
: "${DFINE_ROOT:?Set DFINE_ROOT to the upstream D-FINE repository}"
: "${DFINE_1536_WEIGHT:?Set DFINE_1536_WEIGHT}"

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

python "$PROJECT_ROOT/code/inference/deimv2/infer_deimv2_832.py" \
  --deimv2-root "$DEIMV2_ROOT" \
  --config "$PROJECT_ROOT/configs/transformer/deimv2/deimv2/deimv2_dinov3_s_fisheye.yml" \
  --checkpoint "$DEIMV2_832_WEIGHT" \
  --image-dir "$EVAL_IMAGES" \
  --output "$PROJECT_ROOT/predictions/transformer_inputs/deimv2_832_conf005_iou070_reproduced.json" \
  --conf 0.05 \
  --iou 0.70 \
  --device cuda:0

python "$PROJECT_ROOT/code/inference/dfine/infer_dfine_1536.py" \
  --dfine-root "$DFINE_ROOT" \
  --config "$PROJECT_ROOT/configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1536_v1.yml" \
  --checkpoint "$DFINE_1536_WEIGHT" \
  --image-dir "$EVAL_IMAGES" \
  --output "$PROJECT_ROOT/predictions/transformer_inputs/dfine1536_last_conf005_iou070_reproduced.json" \
  --conf 0.05 \
  --iou 0.70 \
  --device cuda:0

echo "DEIMv2-S-960 prediction JSON is included; its checkpoint is not part of this release."
