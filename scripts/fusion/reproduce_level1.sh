#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${EVAL_IMAGES:?Set EVAL_IMAGES to FishEye1K_eval images}"

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
TMP_DIR="${TMP_DIR:-$PROJECT_ROOT/reproduced}"
mkdir -p "$TMP_DIR"

python "$PROJECT_ROOT/code/fusion/wbf_portable_helper.py" \
  --img-dir "$EVAL_IMAGES" \
  --jsons \
    "$PROJECT_ROOT/predictions/yolo_inputs/yolov10x_second_c050_i065_official.json" \
    "$PROJECT_ROOT/predictions/yolo_inputs/yolov10x_second_c050_i065_s1536_official.json" \
  --weights 1 1 \
  --iou-thr 0.65 \
  --skip-box-thr 0.001 \
  --final-thr 0.001 \
  --topk 300 \
  --out "$TMP_DIR/Y10_MS_1280_1536_reproduced.json"

python "$PROJECT_ROOT/scripts/validation/compare_prediction_json.py" \
  "$PROJECT_ROOT/predictions/intermediate/Y10_MS_1280_1536.json" \
  "$TMP_DIR/Y10_MS_1280_1536_reproduced.json" \
  --decimals 5

python "$PROJECT_ROOT/code/fusion/wbf_portable_helper.py" \
  --img-dir "$EVAL_IMAGES" \
  --jsons \
    "$PROJECT_ROOT/predictions/yolo_inputs/yolov13l_second_c0495_i045_official.json" \
    "$PROJECT_ROOT/predictions/yolo_inputs/yolov13l_second_c0495_i045_s1536_official.json" \
  --weights 1 1 \
  --iou-thr 0.65 \
  --skip-box-thr 0.001 \
  --final-thr 0.001 \
  --topk 300 \
  --out "$TMP_DIR/Y13_MS_1280_1536_reproduced.json"

python "$PROJECT_ROOT/scripts/validation/compare_prediction_json.py" \
  "$PROJECT_ROOT/predictions/intermediate/Y13_MS_1280_1536.json" \
  "$TMP_DIR/Y13_MS_1280_1536_reproduced.json" \
  --decimals 6
