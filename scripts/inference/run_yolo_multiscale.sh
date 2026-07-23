#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"
: "${EVAL_IMAGES:?Set EVAL_IMAGES to FishEye1K_eval images}"
: "${YOLOR_WEIGHT:?Set YOLOR_WEIGHT}"
: "${Y10_WEIGHT:?Set Y10_WEIGHT}"
: "${Y13_WEIGHT:?Set Y13_WEIGHT}"

cd "$PROJECT_ROOT"

python code/inference/yolo/infer_YR.py --image_folder "$EVAL_IMAGES" --yolor_model "$YOLOR_WEIGHT" --output predictions/yolo_inputs/yolor_d6_2026_conf005_iou070_official.json --img_size 1280 --yolor_conf 0.05 --yolor_iou 0.70 --device cuda:0
python code/inference/yolo/infer_Y10.py --image_folder "$EVAL_IMAGES" --yolov10_model "$Y10_WEIGHT" --output predictions/yolo_inputs/yolov10x_second_c050_i065_official.json --img_size 1280 --yolov10_conf 0.50 --yolov10_iou 0.65 --device cuda:0
python code/inference/yolo/infer_Y10.py --image_folder "$EVAL_IMAGES" --yolov10_model "$Y10_WEIGHT" --output predictions/yolo_inputs/yolov10x_second_c050_i065_s1536_official.json --img_size 1536 --yolov10_conf 0.50 --yolov10_iou 0.65 --device cuda:0
python code/inference/yolo/infer_Y13.py --model "$Y13_WEIGHT" --image_dir "$EVAL_IMAGES" --output predictions/yolo_inputs/yolov13l_second_c0495_i045_official.json --img_size 1280 --conf 0.495 --iou 0.45 --device cuda:0
python code/inference/yolo/infer_Y13.py --model "$Y13_WEIGHT" --image_dir "$EVAL_IMAGES" --output predictions/yolo_inputs/yolov13l_second_c0495_i045_s1536_official.json --img_size 1536 --conf 0.495 --iou 0.45 --device cuda:0
