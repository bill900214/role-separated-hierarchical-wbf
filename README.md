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

## Important Scope Statement

The YOLOR-D6, YOLOv10-X, and YOLOv13-L checkpoints are publicly released UT-T1 checkpoints. This work does **not** claim their original training as a contribution. The contributions here are multi-scale inference, output-format unification, hierarchical WBF, day/night class-wise thresholding, Transformer fine-tuning, and controlled ablation.

The reported evaluation uses the 1,000-image FishEye1K_eval set. Its reference annotations remain hidden on the official evaluation server. The repository therefore provides prediction JSON files and server-returned metrics, but not hidden ground-truth annotations.

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

All reported values were returned by the official AI City Challenge evaluation platform for the same FishEye1K_eval submission protocol.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── .gitignore
├── configs/
│   └── fusion/final_msdnl.yaml
├── fusion/
│   └── day_night_classwise_threshold.py
├── legacy/
│   └── wbf_fuse_results_original_found.py
├── scripts/
│   ├── training/
│   ├── inference/
│   └── fusion/
├── predictions/
│   ├── yolo_inputs/
│   ├── transformer_inputs/
│   ├── intermediate/
│   └── final/
├── results/
├── checkpoints/
├── docs/
└── third_party/
```

## Installation

The original detectors use separate upstream repositories and environments. See:

- `docs/YOLO_MULTISCALE.md`
- `docs/DFINE_THREE_STAGE.md`
- `docs/DEIMV2.md`
- `third_party/README.md`

The final JSON conversion and fusion environment used Python 3.11 with PyTorch 2.5.1, torchvision 0.20.1, NumPy 2.4.3, OpenCV 4.13.0.92, and Pillow 12.1.1.

## Prediction Files

The recommended public files include:

```text
predictions/yolo_inputs/
predictions/transformer_inputs/
predictions/intermediate/
predictions/final/FINAL_MSDN_L_EC2.json
```

The final JSON contains model predictions only. It does not contain hidden ground-truth annotations.

## Reproducibility Status

The repository provides:

- fixed prediction JSON inputs;
- intermediate JSON outputs;
- final server-submission JSON;
- final WBF configuration;
- day/night threshold implementation;
- model/checkpoint inventory and checksums where available.

Items still requiring verification before a public release:

- exact DEIMv2-S-960 checkpoint filename and SHA-256;
- exact upstream commit hashes;
- exact original shell commands for every training/inference stage;
- final license compatibility review.

## Citation

See `CITATION.cff`.

## License

A final repository license should be selected only after checking the licenses of all included third-party files. Original code written specifically for this project may be licensed separately from upstream-derived code.
