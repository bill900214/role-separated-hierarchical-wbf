# Role-Separated Hierarchical WBF for Multi-Scale YOLO–Transformer Fusion in Fisheye Road Object Detection

Official research repository for:

> **Role-Separated Hierarchical WBF for Multi-Scale YOLO–Transformer Fusion in Fisheye Road Object Detection**  
> Ding-Jun Huang and Chun-Ming Tsai  
> Department of Computer Science, University of Taipei, Taipei, Taiwan

## Overview

This repository archives the code, configurations, prediction JSON files, intermediate outputs, and official evaluation records used in a rectification-free fisheye road-object detection pipeline.

The method separates three fusion roles:

1. **Level I — Same-model multi-scale fusion**
   - YOLOv10-X: 1280 and 1536 predictions are fused with equal weights.
   - YOLOv13-L: 1280 and 1536 predictions are fused with equal weights.

2. **Level II — YOLO main branch**
   - YOLOR-D6 at 1280
   - multi-scale YOLOv10-X
   - multi-scale YOLOv13-L

3. **Level III — YOLO–Transformer fusion**
   - thresholded YOLO main branch
   - DEIMv2-DINOv3-S at 960
   - DEIMv2-DINOv3-S at 832
   - D-FINE-L at 1536

A dataset-specific Day/Night Class-Wise Confidence Thresholding step is applied to the YOLO main branch before Level III fusion.

## Scope and Checkpoint Provenance

The YOLOR-D6, YOLOv10-X, and YOLOv13-L checkpoints are publicly released UT-T1 checkpoints. Their original training is **not** claimed as a contribution of this study.

The repository documents the work performed for this study:

- multi-scale inference;
- output-format unification;
- hierarchical WBF;
- Day/Night class-wise confidence thresholding;
- D-FINE and DEIMv2 experiment records;
- controlled ablation;
- prediction artifacts used by the official evaluation submission.

The reported evaluation uses the 1,000-image FishEye1K_eval set. Its reference annotations remain hidden on the official evaluation platform. This repository therefore provides prediction JSON files and platform-returned metrics, but not hidden ground-truth annotations.

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

## Official Results

| Method | F1 | AP50–95 | AP50 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| Original heterogeneous baseline | 0.6562 | 0.6050 | 0.8060 | 0.4532 | 0.7325 | 0.6196 |
| Multi-scale YOLO | 0.6596 | 0.6123 | — | 0.4665 | — | — |
| Final MSDNL | **0.6604** | **0.6147** | **0.8220** | **0.4709** | **0.7378** | **0.6214** |

All values were returned by the official AI City Challenge evaluation platform under the same FishEye1K_eval submission protocol.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSING.md
├── LICENSES/
├── checkpoints/
├── code/
│   ├── data_tools/
│   ├── fusion/
│   └── inference/
├── configs/
│   ├── README.md
│   ├── fusion/
│   └── transformer/
├── docs/
├── environment/
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
│   ├── inference/
│   ├── fusion/
│   └── validation/
└── third_party/
```

## Key Documentation

- `docs/YOLO_MULTISCALE.md`: YOLO multi-scale inference and checkpoint provenance.
- `docs/DFINE_THREE_STAGE.md`: retained D-FINE three-stage output records.
- `docs/DEIMV2.md`: DEIMv2 auxiliary branches and prediction-only release scope for 960.
- `docs/WBF_REPRODUCIBILITY.md`: exact reproducibility scope of the fusion implementation.
- `docs/EVALUATION.md`: hidden-test evaluation protocol.
- `docs/PATH_CONFIGURATION.md`: dataset, upstream repository, and checkpoint path configuration.

## Quick Integrity Checks

```bash
python scripts/validation/validate_prediction_json.py \
  predictions/final/FINAL_MSDN_L_EC2.json \
  --expected-images 1000

export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
bash scripts/fusion/reproduce_day_night.sh
```

Level-I inspection additionally requires the FishEye1K_eval image directory:

```bash
export EVAL_IMAGES=/path/to/FishEye1K_eval/images
bash scripts/fusion/reproduce_level1.sh
```

## Prediction Artifacts

Final official-platform submission:

```text
predictions/final/FINAL_MSDN_L_EC2.json
```

DEIMv2-S-960 prediction input used by Level III:

```text
predictions/transformer_inputs/submission_deimv2_960_conf052_iou045.json
```

These JSON files contain model predictions only and do not contain hidden ground-truth annotations. SHA-256 files are stored alongside them.

## Reproducibility Scope

The repository provides the actual committed prediction artifacts used by the reported pipeline.

Verified portable components include:

- JSON format validation;
- Day/Night threshold reconstruction;
- final artifact checksum validation;
- Level-I multi-scale fusion inspection.

The original experiment used the MMDetection implementation of `weighted_boxes_fusion`. The precise original MMDetection/MMEngine environment metadata was not preserved in the collected archive. Therefore, the portable helper under `code/fusion/` is **not claimed to reproduce the original Level-II and Level-III outputs bit-for-bit**.

The committed intermediate and final JSON files are the authoritative archival outputs. See `docs/WBF_REPRODUCIBILITY.md`.

## Checkpoint Availability

- D-FINE-L-1536: retained checkpoint record and checksum.
- DEIMv2-S-832: retained checkpoint record and checksum.
- DEIMv2-S-960: actual prediction JSON released; original checkpoint not redistributed because it was not retained in the collected experiment archive.

The DEIMv2-S-960 branch is therefore documented as **prediction-only** and is not claimed to be checkpoint-level reproducible.

## Citation

See `CITATION.cff`.

## Licensing

This repository contains original project code and adapted files derived from multiple upstream projects. No blanket relicensing of upstream-derived files is asserted. Consult `LICENSING.md`, `third_party/README.md`, and the corresponding upstream licenses before redistribution or reuse.
