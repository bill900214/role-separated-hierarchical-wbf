#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${EVAL_IMAGES:?Set EVAL_IMAGES to FishEye1K_eval images}"

: "${YOLOR_ROOT:?Set YOLOR_ROOT to the upstream YOLOR repository}"
: "${YOLOV10_ROOT:?Set YOLOV10_ROOT to the upstream YOLOv10 repository}"
: "${YOLOV13_ROOT:?Set YOLOV13_ROOT to the upstream YOLOv13 repository}"

: "${YOLOR_WEIGHT:?Set YOLOR_WEIGHT}"
: "${Y10_WEIGHT:?Set Y10_WEIGHT}"
: "${Y13_WEIGHT:?Set Y13_WEIGHT}"

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
mkdir -p "$PROJECT_ROOT/predictions/yolo_inputs"

run_yolor() {
  PYTHONPATH="$YOLOR_ROOT:${PYTHONPATH:-}" \
  python "$PROJECT_ROOT/code/inference/yolo/infer_YR.py" "$@"
}

run_yolov10() {
  PYTHONPATH="$YOLOV10_ROOT:${PYTHONPATH:-}" \
  python "$PROJECT_ROOT/code/inference/yolo/infer_Y10.py" "$@"
}

run_yolov13() {
  PYTHONPATH="$YOLOV13_ROOT:${PYTHONPATH:-}" \
  python "$PROJECT_ROOT/code/inference/yolo/infer_Y13.py" "$@"
}

run_yolor \
  --image_folder "$EVAL_IMAGES" \
  --yolor_model "$YOLOR_WEIGHT" \
  --output "$PROJECT_ROOT/predictions/yolo_inputs/yolor_d6_2026_conf005_iou070_official.json" \
  --img_size 1280 \
  --yolor_conf 0.05 \
  --yolor_iou 0.70 \
  --device cuda:0

run_yolov10 \
  --image_folder "$EVAL_IMAGES" \
  --yolov10_model "$Y10_WEIGHT" \
  --output "$PROJECT_ROOT/predictions/yolo_inputs/yolov10x_second_c050_i065_official.json" \
  --img_size 1280 \
  --yolov10_conf 0.50 \
  --yolov10_iou 0.65 \
  --device cuda:0

run_yolov10 \
  --image_folder "$EVAL_IMAGES" \
  --yolov10_model "$Y10_WEIGHT" \
  --output "$PROJECT_ROOT/predictions/yolo_inputs/yolov10x_second_c050_i065_s1536_official.json" \
  --img_size 1536 \
  --yolov10_conf 0.50 \
  --yolov10_iou 0.65 \
  --device cuda:0

run_yolov13 \
  --model "$Y13_WEIGHT" \
  --image_dir "$EVAL_IMAGES" \
  --output "$PROJECT_ROOT/predictions/yolo_inputs/yolov13l_second_c0495_i045_official.json" \
  --img_size 1280 \
  --conf 0.495 \
  --iou 0.45 \
  --device cuda:0

run_yolov13 \
  --model "$Y13_WEIGHT" \
  --image_dir "$EVAL_IMAGES" \
  --output "$PROJECT_ROOT/predictions/yolo_inputs/yolov13l_second_c0495_i045_s1536_official.json" \
  --img_size 1536 \
  --conf 0.495 \
  --iou 0.45 \
  --device cuda:0
