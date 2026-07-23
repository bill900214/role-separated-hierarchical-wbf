# Official Evaluation Records

`official_metrics.csv` records the complete progressive metrics returned by
the official AI City Challenge evaluation platform.

Included stages:

1. Three-YOLO equal-weight fusion without Day/Night thresholds.
2. Three-YOLO main branch with Day/Night scene-specific class-wise thresholds.
3. Original heterogeneous baseline.
4. Multi-scale YOLO.
5. Final MSDNL.

The complete Multi-scale YOLO row is:

```text
F1 = 0.6596
AP50–95 = 0.6123
AP50 = 0.8170
AP_S = 0.4665
AP_M = 0.7362
AP_L = 0.6214
```

Hidden FishEye1K_eval reference annotations are not included, so official
F1 and AP metrics cannot be recalculated locally.

`artifact_checksums.sha256` records SHA-256 values for retained prediction
artifacts. See `docs/RESULTS_PROGRESSION.md` for the formatted progression.
