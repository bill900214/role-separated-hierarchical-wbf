# Method and Parameters

## Model Roles

| Branch | Model | Inference resolution |
|---|---|---:|
| YOLO main | YOLOR-D6 | 1280 |
| YOLO main | YOLOv10-X | 1280, 1536 |
| YOLO main | YOLOv13-L | 1280, 1536 |
| Transformer auxiliary | DEIMv2-S | 960, 832 |
| Transformer auxiliary | D-FINE-L | 1536 |

The YOLO models use public UT-T1 checkpoints and are used for inference only
in this study. The Transformer models were fine-tuned on the 17,629-image
composite training pool described in the paper.

## Level I — Same-Model Multi-Scale Fusion

YOLOv10-X and YOLOv13-L are each inferred at 1280 and 1536. The two
resolution streams of each model are fused with equal weights.

```text
weights = 1:1
IoU = 0.65
skip threshold = 0.001
output threshold = 0.001
top-k = 300
```

## Level II — YOLO Main Branch

```text
inputs = YOLOR-D6 + YOLOv10-X(MS) + YOLOv13-L(MS)
normalized weights = 1:1:1
stored implementation values = 9:9:9
IoU = 0.65
skip threshold = 0.15
```

## Scene-Adaptive Class-Wise Thresholding

```python
str(image_id).startswith("293")
```

Class order:

```text
Bus, Bike, Car, Pedestrian, Truck
```

Thresholds:

```text
Day:   0.28, 0.28, 0.28, 0.28, 0.23
Night: 0.08, 0.13, 0.18, 0.13, 0.18
```

## Level III — Main–Auxiliary Fusion

```text
inputs =
  thresholded YOLO main branch
  DEIMv2-S-960
  DEIMv2-S-832
  D-FINE-L-1536

weights = 1.2 : 0.065 : 0.05 : 0.05
IoU = 0.65
skip threshold = 0.001
final confidence threshold = 0.295
top-k = 300
```

The complete machine-readable configuration is stored in:

```text
configs/fusion/final_msdnl.yaml
```
