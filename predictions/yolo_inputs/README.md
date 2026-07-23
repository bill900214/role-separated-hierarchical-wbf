# YOLO Prediction Inputs

These files are the detector outputs used to construct the YOLO main branch.

| File | Model | Resolution | Confidence | IoU |
|---|---|---:|---:|---:|
| `yolor_d6_2026_conf005_iou070_official.json` | YOLOR-D6 | 1280 | 0.05 | 0.70 |
| `yolov10x_second_c050_i065_official.json` | YOLOv10-X | 1280 | 0.50 | 0.65 |
| `yolov10x_second_c050_i065_s1536_official.json` | YOLOv10-X | 1536 | 0.50 | 0.65 |
| `yolov13l_second_c0495_i045_official.json` | YOLOv13-L | 1280 | 0.495 | 0.45 |
| `yolov13l_second_c0495_i045_s1536_official.json` | YOLOv13-L | 1536 | 0.495 | 0.45 |

The YOLO checkpoints originate from the public UT-T1 release. Their original training is not claimed as a contribution of this work.
