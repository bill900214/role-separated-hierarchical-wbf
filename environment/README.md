# Environment Records

The detectors were executed in separate upstream environments.

## Known Utility/Fusion Environment Record

A retained conversion and analysis environment used:

```text
Python 3.11
PyTorch 2.5.1
torchvision 0.20.1
NumPy 2.4.3
OpenCV 4.13.0.92
Pillow 12.1.1
```

This record does not prove that the exact original MMDetection WBF execution used the same package set.

## Original MMDetection WBF Environment

The original script imported:

```python
from mmdet.models.utils import weighted_boxes_fusion
```

The exact MMDetection and MMEngine version identifiers were not preserved in the collected archive. Do not invent versions. Recompute Level-II/III only after recovering or selecting a compatible environment and validating outputs against the committed JSON artifacts.

## Upstream Environments

YOLOR, YOLOv10, YOLOv13, D-FINE, and DEIMv2 should be installed according to their corresponding upstream repositories. Their source roots are supplied to the wrappers through environment variables.
