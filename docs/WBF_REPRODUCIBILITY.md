# WBF Reproducibility Scope

## Original Experiment Implementation

The retained original experimental script imports:

```python
from mmdet.models.utils import weighted_boxes_fusion
```

It is preserved at:

```text
legacy/wbf_fuse_results_original_found.py
```

The precise original MMDetection and MMEngine package versions were not retained in the collected experiment archive.

## Portable Inspection Helper

The standalone comparison helper is:

```text
code/fusion/wbf_portable_helper.py
```

This helper is not the original MMDetection implementation.

## Verified Portable Scope

- Day/Night class-wise thresholding exactly reconstructs `MSDN_L_SEC_MS3_DN.json`.
- YOLOv13-L Level-I same-model multi-scale fusion is verified from retained JSON inputs.
- YOLOv10-X Level-I reconstruction matches the retained artifact with numerical tolerance near `1e-6`.

Optional Level-I check:

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
export EVAL_IMAGES=/path/to/FishEye1K_eval/images
bash scripts/fusion/reproduce_level1.sh
```

Exact Day/Night check:

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
bash scripts/fusion/reproduce_day_night.sh
```

## Level-II and Level-III Boundary

The portable helper is not claimed to reconstruct the original Level-II or Level-III outputs bit-for-bit.

Authoritative archival artifacts:

```text
predictions/intermediate/Y10_MS_1280_1536.json
predictions/intermediate/Y13_MS_1280_1536.json
predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json
predictions/intermediate/MSDN_L_SEC_MS3_DN.json
predictions/final/FINAL_MSDN_L_EC2.json
```

Exact end-to-end recomputation should not be claimed unless an MMDetection/MMEngine environment is validated against these retained outputs.
