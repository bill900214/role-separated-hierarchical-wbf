# WBF Reproducibility Scope

## Original Experiment Implementation

The retained original experimental script imports:

```python
from mmdet.models.utils import weighted_boxes_fusion
```

The corresponding script is preserved at:

```text
legacy/wbf_fuse_results_original_found.py
```

It uses the MMDetection WBF implementation and the experiment-specific image list derived from a COCO annotation file.

## Portable Helper

A portable helper is retained at:

```text
code/fusion/wbf_fuse_official.py
```

This helper is useful for format inspection and approximate reconstruction, but it is not the same implementation as the original MMDetection function.

## Verified Scope

The repository audit verified:

- Day/Night class-wise thresholding exactly reconstructs `MSDN_L_SEC_MS3_DN.json`.
- YOLOv13-L Level-I multi-scale fusion can be reconstructed from the committed prediction inputs.
- YOLOv10-X Level-I reconstruction matches the retained detection count, with possible floating-point differences near `1e-6`.

## Unresolved Exact-Reproduction Scope

The precise original MMDetection/MMEngine package versions were not preserved in the collected archive. The current portable helper is therefore **not claimed to reconstruct the original Level-II and Level-III JSON files bit-for-bit**.

The following committed files are the authoritative experimental artifacts:

```text
predictions/intermediate/Y10_MS_1280_1536.json
predictions/intermediate/Y13_MS_1280_1536.json
predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json
predictions/intermediate/MSDN_L_SEC_MS3_DN.json
predictions/final/FINAL_MSDN_L_EC2.json
```

## Recommended Use

Use the repository to:

- inspect the actual input and output prediction artifacts;
- verify JSON integrity and checksums;
- reproduce the Day/Night threshold stage;
- inspect the exact reported fusion parameters;
- compare alternative WBF implementations against the archived outputs.

Do not describe the portable helper as an exact end-to-end reproduction of the original MMDetection Level-II/III pipeline unless the original package versions are recovered and output equivalence is verified.
