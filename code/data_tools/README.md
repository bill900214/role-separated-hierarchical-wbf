# Data Conversion Utilities

The retained utilities support dataset conversion and COCO-style prediction formatting.

```text
dataprocessing/visdrone2yolo.py
dataprocessing/yolo2coco.py
root/convert_visdrone.py
root/merge_yolo_to_coco.py
```

Duplicate copies of `visdrone2yolo.py` and `yolo2coco.py` were removed to avoid ambiguity.

These scripts are dataset-preparation utilities. They are not used to evaluate the hidden FishEye1K_eval ground truth locally.
