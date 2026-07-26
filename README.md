# Role-Separated Hierarchical Fusion for Multi-Scale YOLO–Transformer Fisheye Object Detection

Research code, configurations, and prediction artifacts for the role-separated
hierarchical fusion framework described in the accompanying manuscript.

**Authors:** Chun-Ming Tsai ([ORCID: 0000-0002-9160-3899](https://orcid.org/0000-0002-9160-3899)), Ding-Jun Huang ([ORCID: 0009-0001-8171-9062](https://orcid.org/0009-0001-8171-9062)), Jun-Wei Hsieh, and Ming-Ching Chang

## Overview

The method is a rectification-free fisheye object-detection pipeline with four
ordered operations:

1. **Level I — Same-model multi-scale fusion:** YOLOv10-X and YOLOv13-L
   predictions at 1280 and 1536 are consolidated independently.
2. **Level II — YOLO main-branch fusion:** YOLOR-D6, multi-scale YOLOv10-X,
   and multi-scale YOLOv13-L form a dense consensus branch.
3. **Scene-adaptive class-wise thresholding:** day and night scenes use
   different class-specific operating points.
4. **Level III — YOLO–Transformer fusion:** low-weight DEIMv2-S and D-FINE-L
   predictions refine the thresholded YOLO main branch.

<p align="center">
  <img
    src="docs/assets/role_separated_hierarchical_wbf_pipeline.png"
    alt="Role-separated hierarchical WBF pipeline"
    width="100%"
  >
</p>

## Experimental Setup

| Model | Initialization | Fine-tuning in this study | Inference resolution | Pipeline role |
|---|---|---|---|---|
| YOLOR-D6 | Public UT-T1 checkpoint | None; inference only | 1280 | YOLO anchor |
| YOLOv10-X | Public UT-T1 checkpoint | None; inference only | 1280, 1536 | YOLO anchor and scale diversity |
| YOLOv13-L | Public UT-T1 checkpoint | None; inference only | 1280, 1536 | YOLO anchor and scale diversity |
| DEIMv2-S | Official pretrained checkpoint | 17,629-image pool; 200 epochs; batch 2; AdamW; AMP/EMA | 832, 960 | Query-based auxiliary |
| D-FINE-L | Official pretrained checkpoint | 1280→1440→1536; 60 epochs/stage; batch 1; AdamW | 1536 | Localization auxiliary |

The Transformer fine-tuning pool combines the 5,288-image FishEye8K training
split, class-mapped VisDrone samples, and the public FishEye1K_eval
pseudo-label package described in the manuscript. The 2,712-image FishEye8K
validation split is used for training monitoring and checkpoint selection.

Evaluation uses the 1,000-image FishEye1K_eval server in a post-challenge,
target-domain transductive setting. Server-side ground-truth annotations are
not included in this repository, and the reported result is not presented as
an award-eligible challenge submission.

## Final Configuration

Class order:

```text
Bus, Bike, Car, Pedestrian, Truck
```

### Level I

| Model | Resolutions | Relative weights | IoU | Skip threshold |
|---|---|---:|---:|---:|
| YOLOv10-X | 1280, 1536 | 1:1 | 0.65 | 0.001 |
| YOLOv13-L | 1280, 1536 | 1:1 | 0.65 | 0.001 |

### Level II

```text
Inputs: YOLOR-D6 + YOLOv10-X multi-scale + YOLOv13-L multi-scale
Normalized relative weights: 1:1:1
IoU threshold: 0.65
Skip threshold: 0.15
```

The archived experiment configuration also records the common-scaled vector
`9:9:9`; it expresses the same equal relative weighting.

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
  thresholded YOLO main branch
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

| Configuration | F1 | AP50–95 | AP50 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| YOLO-only main branch | 0.6425 | — | — | — | — | — |
| + Transformer auxiliaries | 0.6562 | 0.6050 | 0.8060 | 0.4532 | 0.7325 | 0.6196 |
| + Same-model multi-scale YOLO | 0.6596 | 0.6123 | 0.8170 | 0.4665 | 0.7362 | 0.6214 |
| Final configuration | **0.6604** | **0.6147** | **0.8220** | **0.4709** | **0.7378** | **0.6214** |

The historical diagnostic stages and Transformer auxiliary ablations reported
in the manuscript are listed in [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository Contents

Included:

- inference and post-processing code;
- final fusion and threshold configurations;
- D-FINE and DEIMv2 configuration records available from the experiments;
- prediction JSON files used by the reported fusion pipeline;
- evaluation-server metrics and artifact checksums.

Not redistributed:

- FishEye8K, FishEye1K_eval, or VisDrone images and annotations;
- hidden evaluation annotations;
- pseudo-label packages;
- public or fine-tuned checkpoint binaries.

See [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md).

## Integrity Check

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

## Documentation

- [`docs/METHOD.md`](docs/METHOD.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md)
- [`docs/SETUP.md`](docs/SETUP.md)
- [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md)

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Author names and the two verified ORCID records are summarized in [`docs/AUTHOR_METADATA.md`](docs/AUTHOR_METADATA.md).

## License

Original project-specific code and documentation are licensed under
[AGPL-3.0](LICENSE). Third-party terms are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
