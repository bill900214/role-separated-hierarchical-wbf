# Method and Experimental Parameters

## Data, Fine-Tuning, and Evaluation

The YOLO branch uses public UT-T1 checkpoints without additional YOLO
fine-tuning in this study. The Transformer auxiliary branch is fine-tuned on a
17,629-image composite pool consisting of the FishEye8K training split,
class-mapped VisDrone samples, and the public FishEye1K_eval pseudo-label
package described in the paper. The 2,712-image FishEye8K validation split is
used for training monitoring and checkpoint selection.

The final system is evaluated on the 1,000-image FishEye1K_eval server. The
experiments use target-domain pseudo-labels and iterative server evaluation;
therefore, they are reported as a post-challenge target-domain transductive
benchmark rather than an award-eligible challenge entry.

## Model Setup

| Model | Initialization | Fine-tuning | Training and inference settings | Role |
|---|---|---|---|---|
| YOLOR-D6 | Public UT-T1 checkpoint | None; inference only | 1280 inference | YOLO anchor |
| YOLOv10-X | Public UT-T1 checkpoint | None; inference only | 1280 and 1536 inference | YOLO anchor and scale diversity |
| YOLOv13-L | Public UT-T1 checkpoint | None; inference only | 1280 and 1536 inference | YOLO anchor and scale diversity |
| DEIMv2-S | Official pretrained checkpoint | 17,629-image pool | batch 2; 200 epochs; AdamW; LR 2e-4; backbone LR 1e-5; AMP/EMA; 832 and 960 inference | Query-based auxiliary |
| D-FINE-L | Official pretrained checkpoint | 17,629-image pool | 1280→1440→1536; batch 1; 60 epochs/stage; AdamW; LR 2.5e-4; backbone LR 1.25e-5; 1536 inference | Localization auxiliary |

## Level I — Same-Model Multi-Scale Fusion

YOLOv10-X and YOLOv13-L are each inferred at 1280 and 1536. The two
resolution streams of each model are fused with equal relative weights.

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
normalized relative weights = 1:1:1
archived common-scaled vector = 9:9:9
IoU = 0.65
skip threshold = 0.15
```

## Scene-Adaptive Class-Wise Thresholding

Night images satisfy:

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

The machine-readable configuration is stored in
`configs/fusion/final_msdnl.yaml`.
