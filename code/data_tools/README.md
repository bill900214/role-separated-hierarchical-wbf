# Data Conversion Utilities

The utilities convert VisDrone-style annotations and YOLO-format labels into
the dataset and COCO-style formats used by the project.

```text
dataprocessing/visdrone2yolo.py
dataprocessing/yolo2coco.py
root/convert_visdrone.py
root/merge_yolo_to_coco.py
```

They do not contain dataset images, hidden annotations, or server-side ground
truth.
