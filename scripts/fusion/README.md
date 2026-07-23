# Fusion and Threshold Scripts

## Level-I Same-Model Multi-Scale Inspection

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
export EVAL_IMAGES=/path/to/FishEye1K_eval/images

bash scripts/fusion/reproduce_level1.sh
```

## Day/Night Threshold Reconstruction

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf

bash scripts/fusion/reproduce_day_night.sh
```

The Day/Night stage is exactly validated against the committed JSON.

## Level-II and Level-III

The original experiment used `mmdet.models.utils.weighted_boxes_fusion`. The exact original package versions were not preserved, so the current portable helper is not claimed to reconstruct Level II or Level III bit-for-bit.

See `docs/WBF_REPRODUCIBILITY.md`.
