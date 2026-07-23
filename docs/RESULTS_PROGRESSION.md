# Official Result Progression

The following values are official hidden-test results returned under the
same 1,000-image FishEye1K_eval submission protocol.

| Stage | Method | F1 | AP50–95 | AP50 | AP_S | AP_M | AP_L |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Three-YOLO equal-weight fusion without Day/Night thresholds | 0.5719 | 0.6360 | 0.8739 | 0.4983 | 0.7426 | 0.6268 |
| 2 | Three-YOLO main branch + Day/Night scene-specific class-wise thresholds | 0.6377 | 0.6033 | 0.8080 | 0.4531 | 0.7284 | 0.6203 |
| 3 | Original heterogeneous baseline | 0.6562 | 0.6050 | 0.8060 | 0.4532 | 0.7325 | 0.6196 |
| 4 | Multi-scale YOLO | 0.6596 | 0.6123 | 0.8170 | 0.4665 | 0.7362 | 0.6214 |
| 5 | Final MSDNL | **0.6604** | **0.6147** | **0.8220** | **0.4709** | **0.7378** | **0.6214** |

## F1 Development

```text
0.5719 → 0.6377 → 0.6562 → 0.6596 → 0.6604
```

## Interpretation Boundary

AP and F1 summarize different aspects of the detector and are not required
to increase monotonically together. The first two rows document the effect
of changing the operating-point thresholding strategy, whereas the later
rows document heterogeneous fusion, multi-scale inference, and the final
role-separated configuration.

## Machine-Readable Record

```text
results/official_metrics.csv
```
