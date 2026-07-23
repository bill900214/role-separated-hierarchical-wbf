# YOLO Main Branch and Multi-Scale Inference

## Checkpoint provenance

YOLOR-D6, YOLOv10-X, and YOLOv13-L use publicly released UT-T1 checkpoints. Their original training is not claimed as a contribution of this work.

## Inference settings

- YOLOR-D6: inference at 1280.
- YOLOv10-X: the same checkpoint is inferred at 1280 and 1536.
- YOLOv13-L: the same checkpoint is inferred at 1280 and 1536.

YOLOv10-X and YOLOv13-L predictions are first fused within each model using equal weights. This avoids assigning two independent votes to the same detector in the final heterogeneous fusion.

## Files to add

```text
scripts/inference/infer_yolor_d6_1280.*
scripts/inference/infer_yolov10_x_1280.*
scripts/inference/infer_yolov10_x_1536.*
scripts/inference/infer_yolov13_l_1280.*
scripts/inference/infer_yolov13_l_1536.*
```

Keep the original script names in a legacy folder if needed, but describe their actual use as inference rather than retraining.
