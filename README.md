# Role-Separated Hierarchical Fusion for Multi-Scale YOLO–Transformer Fisheye Object Detection

Research code, configurations, and prediction artifacts for a
rectification-free fisheye object-detection framework based on
role-separated hierarchical Weighted Boxes Fusion (WBF).

**Authors:** Chun-Ming Tsai, Ding-Jun Huang, Jun-Wei Hsieh, and Ming-Ching Chang

## Overview

The framework separates fusion into three roles:

1. **Level I — Same-model multi-scale fusion:** YOLOv10-X and YOLOv13-L
   predictions at 1280 and 1536 are independently consolidated.
2. **Level II — YOLO main-branch fusion:** YOLOR-D6, multi-scale YOLOv10-X,
   and multi-scale YOLOv13-L are fused with equal relative weights.
3. **Level III — YOLO–Transformer fusion:** the thresholded YOLO main branch
   is combined with low-weight DEIMv2-S and D-FINE-L auxiliary predictions.

<p align="center">
  <img
    src="docs/assets/role_separated_hierarchical_wbf_pipeline.png"
    alt="Role-separated hierarchical WBF pipeline"
    width="100%"
  >
</p>

## Final Configuration

Class order:

```text
Bus, Bike, Car, Pedestrian, Truck
```

### Level I

| Model | Resolutions | Weights | IoU |
|---|---|---:|---:|
| YOLOv10-X | 1280, 1536 | 1:1 | 0.65 |
| YOLOv13-L | 1280, 1536 | 1:1 | 0.65 |

### Level II

```text
Inputs: YOLOR-D6 + YOLOv10-X multi-scale + YOLOv13-L multi-scale
Relative weights: 1:1:1
IoU threshold: 0.65
Skip threshold: 0.15
```

The retained configuration stores the equivalent common-scaled values
`9:9:9`.

### Scene-Adaptive Class-Wise Thresholding

Night images are identified by:

```python
str(image_id).startswith("293")
```

```text
Day:   0.28, 0.28, 0.28, 0.28, 0.23
Night: 0.08, 0.13, 0.18, 0.13, 0.18
```

### Level III

```text
Inputs:
  YOLO main branch
  DEIMv2-S-960
  DEIMv2-S-832
  D-FINE-L-1536

Weights: 1.2 : 0.065 : 0.05 : 0.05
IoU threshold: 0.65
Skip threshold: 0.001
Final confidence threshold: 0.295
Maximum detections per image: 300
```

## Main Results

| Configuration | F1 | AP50–95 | AP50 | AP_S |
|---|---:|---:|---:|---:|
| YOLO-only main branch | 0.6425 | — | — | — |
| + Transformer auxiliaries | 0.6562 | 0.6050 | 0.8060 | 0.4532 |
| + Same-model multi-scale YOLO | 0.6596 | 0.6123 | 0.8170 | 0.4665 |
| Final configuration | **0.6604** | **0.6147** | **0.8220** | **0.4709** |

The evaluation was conducted on the 1,000-image FishEye1K_eval server in
a post-challenge target-domain transductive setting. It is not presented
as an official award-eligible challenge submission.

Full progressive and auxiliary-branch results are available in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Released Contents

This public release contains:

- inference and post-processing code;
- final fusion and threshold configurations;
- D-FINE and DEIMv2 configuration records retained from the experiments;
- model prediction JSON files used in the reported fusion pipeline;
- official evaluation metrics and artifact checksums.

This repository does **not** redistribute:

- FishEye8K, FishEye1K_eval, or VisDrone images;
- hidden evaluation annotations;
- pseudo-label packages;
- third-party or fine-tuned checkpoint binaries.

See [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md).

## Quick Integrity Check

```bash
python -m pip install -r requirements.txt
python scripts/validation/verify_repository.py
```

Expected final lines:

```text
Final prediction: 33834 detections / 1000 images
Day/Night reconstruction: MATCH
Repository status: PASS
```

## Repository Structure

```text
.
├── code/
├── configs/
├── docs/
├── fusion/
├── predictions/
│   ├── yolo_inputs/
│   ├── transformer_inputs/
│   ├── intermediate/
│   ├── ablation/
│   └── final/
├── results/
├── scripts/
├── CITATION.cff
├── LICENSE
├── NOTICE
└── THIRD_PARTY_NOTICES.md
```

## Documentation

- [`docs/METHOD.md`](docs/METHOD.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md)
- [`docs/SETUP.md`](docs/SETUP.md)
- [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md)

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Original project-specific code and documentation are licensed under
[AGPL-3.0](LICENSE). Third-party terms are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
