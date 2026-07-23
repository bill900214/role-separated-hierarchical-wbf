# YOLO Main Branch and Multi-Scale Inference

## Checkpoint provenance

YOLOR-D6, YOLOv10-X, and YOLOv13-L use publicly released UT-T1 checkpoints. Their original training is not claimed as a contribution of this work.

## Final inference settings

| Model | Resolution | Confidence | IoU |
|---|---:|---:|---:|
| YOLOR-D6 | 1280 | 0.05 | 0.70 |
| YOLOv10-X | 1280 | 0.50 | 0.65 |
| YOLOv10-X | 1536 | 0.50 | 0.65 |
| YOLOv13-L | 1280 | 0.495 | 0.45 |
| YOLOv13-L | 1536 | 0.495 | 0.45 |

YOLOv10-X and YOLOv13-L use the same checkpoint at both resolutions. Their two prediction files are fused within each model using equal weights before entering the heterogeneous YOLO main branch.

## Available inference implementations

```text
code/inference/yolo/infer_YR.py
code/inference/yolo/infer_Y10.py
code/inference/yolo/infer_Y13.py
```

These scripts must be executed in environments containing their corresponding upstream implementations and dependencies.

## Commands

Set paths first:

```bash
export EVAL_IMAGES=/path/to/FishEye1K_eval/images
export YOLOR_WEIGHT=/path/to/yolor_d6_checkpoint.pt
export Y10_WEIGHT=/path/to/yolov10_x_checkpoint.pt
export Y13_WEIGHT=/path/to/yolov13_l_checkpoint.pt
```

### YOLOR-D6 at 1280

```bash
python code/inference/yolo/infer_YR.py \
  --image_folder "$EVAL_IMAGES" \
  --yolor_model "$YOLOR_WEIGHT" \
  --output predictions/yolo_inputs/yolor_d6_2026_conf005_iou070_official.json \
  --img_size 1280 \
  --yolor_conf 0.05 \
  --yolor_iou 0.70 \
  --device cuda:0
```

### YOLOv10-X at 1280

```bash
python code/inference/yolo/infer_Y10.py \
  --image_folder "$EVAL_IMAGES" \
  --yolov10_model "$Y10_WEIGHT" \
  --output predictions/yolo_inputs/yolov10x_second_c050_i065_official.json \
  --img_size 1280 \
  --yolov10_conf 0.50 \
  --yolov10_iou 0.65 \
  --device cuda:0
```

### YOLOv10-X at 1536

```bash
python code/inference/yolo/infer_Y10.py \
  --image_folder "$EVAL_IMAGES" \
  --yolov10_model "$Y10_WEIGHT" \
  --output predictions/yolo_inputs/yolov10x_second_c050_i065_s1536_official.json \
  --img_size 1536 \
  --yolov10_conf 0.50 \
  --yolov10_iou 0.65 \
  --device cuda:0
```

### YOLOv13-L at 1280

```bash
python code/inference/yolo/infer_Y13.py \
  --model "$Y13_WEIGHT" \
  --image_dir "$EVAL_IMAGES" \
  --output predictions/yolo_inputs/yolov13l_second_c0495_i045_official.json \
  --img_size 1280 \
  --conf 0.495 \
  --iou 0.45 \
  --device cuda:0
```

### YOLOv13-L at 1536

```bash
python code/inference/yolo/infer_Y13.py \
  --model "$Y13_WEIGHT" \
  --image_dir "$EVAL_IMAGES" \
  --output predictions/yolo_inputs/yolov13l_second_c0495_i045_s1536_official.json \
  --img_size 1536 \
  --conf 0.495 \
  --iou 0.45 \
  --device cuda:0
```
