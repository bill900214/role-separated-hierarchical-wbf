# Environment Records

The detectors were executed in separate upstream environments.

## Portable Utilities

The local validators and Day/Night threshold utility require Python 3.10 or newer.

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

The precise MMDetection and MMEngine version identifiers were not preserved. No version is invented in this release.

## Upstream Environments

Install YOLOR, YOLOv10, YOLOv13, D-FINE, and DEIMv2 according to their upstream repositories. Source roots are supplied to the wrappers through environment variables.
