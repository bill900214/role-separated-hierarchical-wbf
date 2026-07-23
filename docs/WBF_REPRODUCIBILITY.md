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

The precise original MMDetection/MMEngine package versions were not preserved in the collected archive.

## Portable Helper

The standalone comparison helper is:

```text
code/fusion/wbf_portable_helper.py
```

A compatibility entry point remains at:

```text
code/fusion/wbf_fuse_official.py
```

Despite the historical filename, the compatibility entry point is **not** the original MMDetection implementation.

## Verified Portable Scope

- The Day/Night class-wise stage exactly reconstructs `MSDN_L_SEC_MS3_DN.json`.
- YOLOv13-L Level-I same-model multi-scale fusion can be reconstructed from the retained JSON inputs.
- YOLOv10-X Level-I reconstruction matches the retained artifact when compared at five decimal places; differences near `1e-6` may occur across numerical environments.

Run:

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
export EVAL_IMAGES=/path/to/FishEye1K_eval/images

bash scripts/fusion/reproduce_level1.sh
bash scripts/fusion/reproduce_day_night.sh
```

## Level-II and Level-III Scope

The portable helper is not claimed to reconstruct the original Level-II or Level-III outputs bit-for-bit.

The authoritative archival artifacts are:

```text
predictions/intermediate/Y10_MS_1280_1536.json
predictions/intermediate/Y13_MS_1280_1536.json
predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json
predictions/intermediate/MSDN_L_SEC_MS3_DN.json
predictions/final/FINAL_MSDN_L_EC2.json
```

Do not claim exact end-to-end recomputation until an MMDetection/MMEngine environment has been validated against these committed outputs.
