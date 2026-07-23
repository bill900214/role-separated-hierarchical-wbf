# Role-Separated Hierarchical WBF for Multi-Scale YOLO–Transformer Fusion in Fisheye Road Object Detection

[![Artifact integrity](https://github.com/bill900214/role-separated-hierarchical-wbf/actions/workflows/artifact-integrity.yml/badge.svg)](https://github.com/bill900214/role-separated-hierarchical-wbf/actions/workflows/artifact-integrity.yml)

Companion artifact repository for a manuscript prepared for the **AI City Challenge Workshop at ECCV 2026**.

> **Role-Separated Hierarchical WBF for Multi-Scale YOLO–Transformer Fusion in Fisheye Road Object Detection**  
> Ding-Jun Huang and Chun-Ming Tsai  
> Department of Computer Science, University of Taipei, Taipei, Taiwan

## Highlights

- Rectification-free fisheye road-object detection.
- Same-model multi-scale fusion for YOLOv10-X and YOLOv13-L.
- A YOLO main branch combining YOLOR-D6, YOLOv10-X, and YOLOv13-L.
- Dataset-specific Day/Night Class-Wise Confidence Thresholding.
- Low-weight Transformer auxiliary predictions from DEIMv2 and D-FINE.
- Final official hidden-test result: **F1 = 0.6604**.

## Method at a Glance

```mermaid
flowchart LR
    Y10A["YOLOv10-X<br/>1280"] --> L1Y10["Level I WBF<br/>1:1"]
    Y10B["YOLOv10-X<br/>1536"] --> L1Y10

    Y13A["YOLOv13-L<br/>1280"] --> L1Y13["Level I WBF<br/>1:1"]
    Y13B["YOLOv13-L<br/>1536"] --> L1Y13

    YR["YOLOR-D6<br/>1280"] --> L2["Level II YOLO WBF<br/>9:9:9"]
    L1Y10 --> L2
    L1Y13 --> L2

    L2 --> DN["Day/Night<br/>Class-Wise Thresholding"]

    DN --> L3["Level III WBF"]
    D960["DEIMv2-S<br/>960"] --> L3
    D832["DEIMv2-S<br/>832"] --> L3
    DF["D-FINE-L<br/>1536"] --> L3

    L3 --> FINAL["FINAL_MSDN_L_EC2.json"]
```

## Final Configuration

Class order:

```text
Bus, Bike, Car, Pedestrian, Truck
```

### Level I — Same-Model Multi-Scale WBF

| Input | Weights | IoU | Skip threshold | Output threshold |
|---|---:|---:|---:|---:|
| YOLOv10-X 1280 + 1536 | 1:1 | 0.65 | 0.001 | 0.001 |
| YOLOv13-L 1280 + 1536 | 1:1 | 0.65 | 0.001 | 0.001 |

### Level II — YOLO Main Branch

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

### Level III — Final YOLO–Transformer WBF

| Input | Weight |
|---|---:|
| Thresholded YOLO main branch | 1.2 |
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

The values above were returned by the official AI City Challenge evaluation platform under the same 1,000-image FishEye1K_eval submission protocol.

## Five-Minute Artifact Check

```bash
python -m pip install -r requirements.txt
python scripts/validation/verify_repository.py
```

Expected key outputs:

```text
Final prediction: 33,834 detections / 1,000 images
DEIMv2-S-960: 30,460 detections / 1,000 images
Day/Night reconstruction: MATCH
Repository status: PASS
```

See [`docs/REVIEWER_QUICKSTART.md`](docs/REVIEWER_QUICKSTART.md) for the reviewer-oriented verification path.

## Artifact Map

| Method stage | Inputs | Archived output | Main implementation/document |
|---|---|---|---|
| Level I YOLOv10-X | 1280 and 1536 JSON | `Y10_MS_1280_1536.json` | `scripts/fusion/reproduce_level1.sh` |
| Level I YOLOv13-L | 1280 and 1536 JSON | `Y13_MS_1280_1536.json` | `scripts/fusion/reproduce_level1.sh` |
| Level II YOLO main | YOLOR-D6, Y10_MS, Y13_MS | `SEC_MS3_Y10Y13MS_raw.json` | archival MMDetection WBF record |
| Day/Night threshold | YOLO main raw JSON | `MSDN_L_SEC_MS3_DN.json` | `fusion/day_night_classwise_threshold.py` |
| Level III final | YOLO main + three Transformer inputs | `FINAL_MSDN_L_EC2.json` | archival official-submission artifact |

A detailed mapping is provided in [`docs/METHOD_ARTIFACT_MAP.md`](docs/METHOD_ARTIFACT_MAP.md).

A complete hash and detection-count table is provided in [`docs/ARTIFACT_MANIFEST.md`](docs/ARTIFACT_MANIFEST.md).

## Reproducibility Status

| Scope | Status |
|---|---|
| Prediction JSON schema and integrity | Verified |
| Final and DEIMv2-S-960 SHA-256 | Verified |
| Day/Night threshold stage | Exact reconstruction |
| Level I YOLOv13-L | Verified |
| Level I YOLOv10-X | Verified with numerical tolerance near 1e-6 |
| Level II and Level III bit-for-bit recomputation | Archival outputs provided; exact original MMDetection/MMEngine versions were not retained |

The original Level II and Level III experiments used `mmdet.models.utils.weighted_boxes_fusion`. The standalone helper in this repository is not represented as a bit-for-bit replacement for that retained experimental environment.

See [`docs/WBF_REPRODUCIBILITY.md`](docs/WBF_REPRODUCIBILITY.md) and [`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md).

## Data and Checkpoints

- FishEye1K_eval images and hidden reference annotations are not redistributed.
- YOLOR-D6, YOLOv10-X, and YOLOv13-L use public UT-T1 checkpoints; their original training is not claimed as a contribution.
- D-FINE-L-1536 and DEIMv2-S-832 checkpoint records are documented, but binaries are not bundled in normal Git history.
- DEIMv2-S-960 is released as the actual prediction JSON used by Level III; the original checkpoint was not retained in the collected experiment archive.

See [`checkpoints/README.md`](checkpoints/README.md), [`docs/DFINE_THREE_STAGE.md`](docs/DFINE_THREE_STAGE.md), and [`docs/DEIMV2.md`](docs/DEIMV2.md).

## Repository Layout

```text
.
├── .github/workflows/
├── checkpoints/
├── code/
├── configs/
├── docs/
├── environment/
├── fusion/
├── legacy/
├── predictions/
├── results/
├── scripts/
├── third_party/
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Third-Party and Reuse Notice

This repository does not apply one blanket license to all files. Upstream licenses remain applicable to upstream-derived code, configurations, and dependencies. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`docs/LICENSING_AND_REUSE.md`](docs/LICENSING_AND_REUSE.md), and [`third_party/README.md`](third_party/README.md).
