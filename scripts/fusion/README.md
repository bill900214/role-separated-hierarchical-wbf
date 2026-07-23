# Fusion Scripts

## Reproducible Day/Night Stage

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
bash scripts/fusion/reproduce_day_night.sh
```

The script reconstructs the final relaxed Day/Night threshold result and compares it with the committed archival JSON.

## WBF Scope

The original Level-II and Level-III experiments used the MMDetection implementation imported as:

```python
from mmdet.models.utils import weighted_boxes_fusion
```

The precise original package versions were not preserved. The current portable WBF helper is therefore not presented as a bit-for-bit reproduction of the original Level-II/III outputs.

See `docs/WBF_REPRODUCIBILITY.md`.
