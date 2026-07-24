# Official Result Progression

The following values align with Table 3 and the corresponding result
discussion in the manuscript.

| Stage | Configuration | F1 | AP50–95 | AP50 | AP_S | AP_M | AP_L |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Historical Raw Equal-Weight WBF† | 0.5719 | 0.6360 | 0.8739 | 0.4983 | 0.7426 | 0.6268 |
| 2 | Historical Day/Night Thresholding† | 0.6377 | 0.6033 | 0.8080 | 0.4531 | 0.7284 | 0.6203 |
| 3 | Best Recorded YOLO-Only Stage (IoU = 0.6575) | 0.6425 | — | — | — | — | — |
| 4 | Selected YOLO–Transformer Setting | 0.6562 | 0.6050 | 0.8060 | 0.4532 | 0.7325 | 0.6196 |
| 5 | + Same-Model Multi-Scale YOLO | 0.6596 | 0.6123 | 0.8170 | 0.4665 | 0.7362 | 0.6214 |
| 6 | + Final Scene-Specific Thresholding | **0.6604** | **0.6147** | **0.8220** | **0.4709** | **0.7378** | **0.6214** |

† Historical diagnostic stages from early pipeline prototyping. The exact
historical threshold vector associated with the 0.6377 result was not
retained.

## F1 Development

```text
0.5719 → 0.6377 → 0.6425 → 0.6562 → 0.6596 → 0.6604
```

## Important Distinction

The historical 0.6377 operating point is not asserted to use the final
retained Day/Night threshold vectors. The final vectors are:

```text
Day:   0.28, 0.28, 0.28, 0.28, 0.23
Night: 0.08, 0.13, 0.18, 0.13, 0.18
```

These final vectors correspond to the peak 0.6604 configuration.
