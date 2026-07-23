# Role-Separated Hierarchical WBF for Multi-Scale YOLO–Transformer Fusion in Fisheye Road Object Detection

Official research repository for:

> **Role-Separated Hierarchical WBF for Multi-Scale YOLO–Transformer Fusion in Fisheye Road Object Detection**  
> Ding-Jun Huang and Chun-Ming Tsai  
> Department of Computer Science, University of Taipei, Taipei, Taiwan

## Overview

This repository documents a rectification-free, role-separated hierarchical Weighted Boxes Fusion (WBF) pipeline for fisheye road-object detection.

The pipeline separates three fusion roles:

1. **Level I — Same-model multi-scale fusion**
   - YOLOv10-X: 1280 and 1536 predictions are fused with equal weights.
   - YOLOv13-L: 1280 and 1536 predictions are fused with equal weights.

2. **Level II — YOLO main branch**
   - YOLOR-D6 at 1280
   - multi-scale YOLOv10-X
   - multi-scale YOLOv13-L

3. **Level III — YOLO–Transformer fusion**
   - YOLO main branch
   - DEIMv2-DINOv3-S at 960
   - DEIMv2-DINOv3-S at 832
   - D-FINE-L at 1536

A dataset-specific Day/Night Class-Wise Confidence Thresholding step is applied to the YOLO main branch before Level III WBF.

## Scope and Checkpoint Provenance

The YOLOR-D6, YOLOv10-X, and YOLOv13-L checkpoints are publicly released UT-T1 checkpoints. Their original training is not claimed as a contribution of this work. The repository documents the multi-scale inference, output-format unification, hierarchical WBF, Day/Night class-wise thresholding, Transformer fine-tuning, and controlled ablation used in this study.

The reported evaluation uses the 1,000-image FishEye1K_eval set. Its reference annotations remain hidden on the official evaluation server. This repository therefore provides prediction JSON files and server-returned metrics, but not hidden ground-truth annotations.

## Final Configuration

Class order:

```text
Bus, Bike, Car, Pedestrian, Truck
```

### Level I: Same-model multi-scale WBF

| Input | Weights | IoU | Skip threshold | Output threshold |
|---|---:|---:|---:|---:|
| YOLOv10-X 1280 + 1536 | 1:1 | 0.65 | 0.001 | 0.001 |
| YOLOv13-L 1280 + 1536 | 1:1 | 0.65 | 0.001 | 0.001 |

### Level II: YOLO Main Branch

| Inputs | Weights | IoU | Skip threshold | Output threshold |
|---|---:|---:|---:|---:|
| YOLOR-D6 1280 + Y10_MS + Y13_MS | 9:9:9 | 0.65 | 0.15 | 0.001 |

### Day/Night Class-Wise Thresholding

Night images are identified by:

```python
str(image_id).startswith("293")
```

Threshold order: `Bus, Bike, Car, Pedestrian, Truck`

```text
Day:   0.28, 0.28, 0.28, 0.28, 0.23
Night: 0.08, 0.13, 0.18, 0.13, 0.18
```

### Level III: Final YOLO–Transformer WBF

| Input | Weight |
|---|---:|
| YOLO main branch | 1.2 |
| DEIMv2-S-960 | 0.065 |
| DEIMv2-S-832 | 0.05 |
| D-FINE-L-1536 | 0.05 |

```text
IoU threshold: 0.65
Skip threshold: 0.001
Final confidence threshold: 0.295
Maximum detections per image: 300
```

## Results

| Method | F1 | AP50–95 | AP50 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| Original heterogeneous baseline | 0.6562 | 0.6050 | 0.8060 | 0.4532 | 0.7325 | 0.6196 |
| Multi-scale YOLO | 0.6596 | 0.6123 | — | 0.4665 | — | — |
| Final MSDNL | **0.6604** | **0.6147** | **0.8220** | **0.4709** | **0.7378** | **0.6214** |

All reported values were returned by the official AI City Challenge evaluation platform under the same FishEye1K_eval submission protocol.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── .gitignore
├── checkpoints/
├── code/
│   ├── data_tools/
│   ├── fusion/
│   └── inference/
├── configs/
│   ├── fusion/
│   └── transformer/
├── docs/
├── fusion/
├── legacy/
├── predictions/
│   ├── yolo_inputs/
│   ├── transformer_inputs/
│   ├── intermediate/
│   └── final/
├── results/
├── scripts/
│   ├── training/
│   └── inference/
└── third_party/
```

## Documentation

- `docs/YOLO_MULTISCALE.md`: YOLO multi-scale inference and commands.
- `docs/DFINE_THREE_STAGE.md`: D-FINE three-stage fine-tuning.
- `docs/DEIMV2.md`: DEIMv2 auxiliary branches and release scope.
- `docs/EVALUATION.md`: hidden-test evaluation protocol.
- `docs/PATH_CONFIGURATION.md`: dataset and checkpoint path configuration.

## Prediction Files

The final official-platform submission is:

```text
predictions/final/FINAL_MSDN_L_EC2.json
```

The file contains model predictions only and does not contain hidden ground-truth annotations. Its checksum is stored alongside it in `FINAL_MSDN_L_EC2.sha256`.

## Reproducibility Notes

The repository provides fixed input JSON files, intermediate outputs, the final server-submission JSON, final fusion parameters, Day/Night threshold code, model configuration files, and command wrappers.

The DEIMv2-S-960 prediction JSON used by the final fusion is released. The original DEIMv2-S-960 checkpoint filename was not preserved in the collected experiment archive and is therefore not claimed as redistributed here.

## Citation

See `CITATION.cff`.

## Licensing

This repository contains original project code and adapted files derived from multiple upstream projects. Consult `third_party/README.md` and each upstream license before redistribution or reuse.
