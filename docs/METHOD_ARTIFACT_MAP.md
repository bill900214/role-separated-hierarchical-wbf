# Method-to-Artifact Map

## Level I — Same-Model Multi-Scale Fusion

### YOLOv10-X

Inputs:

```text
predictions/yolo_inputs/yolov10x_second_c050_i065_official.json
predictions/yolo_inputs/yolov10x_second_c050_i065_s1536_official.json
```

Archived output:

```text
predictions/intermediate/Y10_MS_1280_1536.json
```

Parameters:

```text
weights = 1:1
IoU = 0.65
skip threshold = 0.001
output threshold = 0.001
```

### YOLOv13-L

Inputs:

```text
predictions/yolo_inputs/yolov13l_second_c0495_i045_official.json
predictions/yolo_inputs/yolov13l_second_c0495_i045_s1536_official.json
```

Archived output:

```text
predictions/intermediate/Y13_MS_1280_1536.json
```

Parameters are identical to the YOLOv10-X Level-I fusion.

## Level II — YOLO Main Branch

Inputs:

```text
predictions/yolo_inputs/yolor_d6_2026_conf005_iou070_official.json
predictions/intermediate/Y10_MS_1280_1536.json
predictions/intermediate/Y13_MS_1280_1536.json
```

Archived output:

```text
predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json
```

Parameters:

```text
weights = 9:9:9
IoU = 0.65
skip threshold = 0.15
output threshold = 0.001
```

The original experiment used the MMDetection WBF implementation. Exact bit-for-bit portable recomputation is not claimed because the original MMDetection/MMEngine version metadata was not retained.

## Day/Night Class-Wise Thresholding

Input:

```text
predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json
```

Exact implementation:

```text
fusion/day_night_classwise_threshold.py
```

Archived output:

```text
predictions/intermediate/MSDN_L_SEC_MS3_DN.json
```

## Level III — Final YOLO–Transformer Fusion

Inputs:

```text
predictions/intermediate/MSDN_L_SEC_MS3_DN.json
predictions/transformer_inputs/submission_deimv2_960_conf052_iou045.json
predictions/transformer_inputs/deimv2_832_conf005_iou070_official.json
predictions/transformer_inputs/dfine1536_last_conf005_iou070_official.json
```

Archived final output:

```text
predictions/final/FINAL_MSDN_L_EC2.json
```

Parameters:

```text
weights = 1.2 : 0.065 : 0.05 : 0.05
IoU = 0.65
skip threshold = 0.001
final confidence threshold = 0.295
top-k = 300 per image
```

The committed final JSON is the actual official-platform submission artifact.
