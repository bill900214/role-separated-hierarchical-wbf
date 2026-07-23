# YOLO Main Branch and Multi-Scale Inference

## Checkpoint Provenance

YOLOR-D6, YOLOv10-X, and YOLOv13-L use publicly released UT-T1 checkpoints. Their original training is not claimed as a contribution of this work.

## Final Inference Settings

| Model | Resolution | Confidence | IoU |
|---|---:|---:|---:|
| YOLOR-D6 | 1280 | 0.05 | 0.70 |
| YOLOv10-X | 1280 | 0.50 | 0.65 |
| YOLOv10-X | 1536 | 0.50 | 0.65 |
| YOLOv13-L | 1280 | 0.495 | 0.45 |
| YOLOv13-L | 1536 | 0.495 | 0.45 |

YOLOv10-X and YOLOv13-L use the same checkpoint at both resolutions. Their two prediction files are fused within each model using equal weights before entering the heterogeneous YOLO main branch.

## Inference Implementations

```text
code/inference/yolo/infer_YR.py
code/inference/yolo/infer_Y10.py
code/inference/yolo/infer_Y13.py
```

`infer_YR.py` requires the upstream YOLOR source tree. The YOLOv10 wrapper contains a local letterbox implementation and uses the `ultralytics` package. YOLOv13 execution requires a compatible upstream environment for the released checkpoint.

## Wrapper

Set paths:

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
export EVAL_IMAGES=/path/to/FishEye1K_eval/images

export YOLOR_ROOT=/path/to/yolor
export YOLOV10_ROOT=/path/to/yolov10
export YOLOV13_ROOT=/path/to/yolov13

export YOLOR_WEIGHT=/path/to/yolor_d6_checkpoint.pt
export Y10_WEIGHT=/path/to/yolov10_x_checkpoint.pt
export Y13_WEIGHT=/path/to/yolov13_l_checkpoint.pt
```

Run:

```bash
bash scripts/inference/run_yolo_multiscale.sh
```

The committed prediction JSON files are the authoritative experiment outputs.
