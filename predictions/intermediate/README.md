# Intermediate Prediction Files

All files contain model predictions in COCO-style JSON format.

## YOLO Main Branch

- `SEC_MS3_Y10Y13MS_raw.json`  
  YOLO Main Branch before Day/Night Class-Wise Thresholding.

- `SEC_MS3_Y10Y13MS_DN.json`  
  Legacy/original Day/Night threshold result using:

  - Day: 0.30, 0.30, 0.30, 0.30, 0.25
  - Night: 0.10, 0.15, 0.20, 0.15, 0.20

- `MSDN_L_SEC_MS3_DN.json`  
  Final relaxed Day/Night threshold result using:

  - Day: 0.28, 0.28, 0.28, 0.28, 0.23
  - Night: 0.08, 0.13, 0.18, 0.13, 0.18

Night images are identified using:

```python
str(image_id).startswith("293")
```

Class order:

```text
Bus, Bike, Car, Pedestrian, Truck
```

## Multi-Scale YOLO

- `Y10_MS_1280_1536.json`  
  Same-model multi-scale WBF result obtained by fusing YOLOv10-X predictions at 1280 × 1280 and 1536 × 1536.

- `Y13_MS_1280_1536.json`  
  Same-model multi-scale WBF result obtained by fusing YOLOv13-L predictions at 1280 × 1280 and 1536 × 1536.

YOLOR-D6 is evaluated at 1280 × 1280 and is combined with the two multi-scale YOLO outputs to form the YOLO Main Branch.

## Level III Ablation

- `A5_SEC_DEIM832.json`  
  YOLO Main Branch combined with DEIMv2-S-832.

- `13_SEC_D832_D960.json`  
  YOLO Main Branch combined with DEIMv2-S-832 and DEIMv2-S-960.

- `14_SEC_D832_D960_DF1536.json`  
  YOLO Main Branch combined with DEIMv2-S-832, DEIMv2-S-960, and D-FINE-L-1536 before the final configuration is applied.

These files are retained to document the progressive contribution of the Transformer auxiliary prediction sources.

## Final YOLO-Side Input

`MSDN_L_SEC_MS3_DN.json` is the final thresholded YOLO-side input used by the final Level III YOLO–Transformer WBF pipeline.

It is not the final submission result. The final submission prediction is stored separately at:

```text
predictions/final/FINAL_MSDN_L_EC2.json
```