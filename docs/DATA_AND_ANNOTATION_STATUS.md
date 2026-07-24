# Data and Annotation Release Status

## Not Redistributed

- FishEye8K/FishEye1K_eval images;
- hidden evaluation annotations;
- third-party dataset images;
- checkpoint binaries.

## Retained Utilities

```text
code/data_tools/dataprocessing/visdrone2yolo.py
code/data_tools/dataprocessing/yolo2coco.py
code/data_tools/root/convert_visdrone.py
code/data_tools/root/merge_yolo_to_coco.py
```

## Current Limitation

The exact merged training annotation package and any retained pseudo-label
files are not bundled in this repository.

This does not affect inspection of the archived prediction outputs, but it
prevents full training-data reconstruction from this repository alone.

For challenge-award reproduction, release the exact redistributable
annotation files or a deterministic script that recreates them from
publicly accessible source annotations.
