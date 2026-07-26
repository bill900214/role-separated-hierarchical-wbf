# Experimental Results

## Main Method Progression

| Stage | Configuration | F1 | AP50–95 | AP50 | AP_S | AP_M | AP_L |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Historical raw equal-weight WBF | 0.5719 | 0.6360 | 0.8739 | 0.4983 | 0.7426 | 0.6268 |
| 2 | Historical Day/Night thresholding | 0.6377 | 0.6033 | 0.8080 | 0.4531 | 0.7284 | 0.6203 |
| 3 | Best recorded YOLO-only stage | 0.6425 | — | — | — | — | — |
| 4 | Selected YOLO–Transformer setting | 0.6562 | 0.6050 | 0.8060 | 0.4532 | 0.7325 | 0.6196 |
| 5 | + Same-model multi-scale YOLO | 0.6596 | 0.6123 | 0.8170 | 0.4665 | 0.7362 | 0.6214 |
| 6 | + Final scene-specific thresholding | **0.6604** | **0.6147** | **0.8220** | **0.4709** | **0.7378** | **0.6214** |

The first two rows are historical diagnostic operating points. The exact
threshold vector for the 0.6377 stage was not preserved; it must not be
interpreted as the final threshold vector. The final Day/Night vectors
correspond to the peak 0.6604 configuration.

## Transformer Auxiliary-Branch Ablation

| Auxiliary configuration | F1 | AP50–95 | AP_S |
|---|---:|---:|---:|
| DEIMv2-S-960 | 0.6528 | 0.5996 | 0.4462 |
| DEIMv2-S-832 | 0.6524 | 0.5968 | 0.4477 |
| D-FINE-L-1536 | 0.6520 | 0.5968 | 0.4483 |
| DEIMv2-S-960/832 + D-FINE-L-1536 | 0.6558 | 0.6035 | 0.4518 |
| Increased DEIMv2-S-960 weight | 0.6560 | 0.6044 | 0.4524 |
| Increased D-FINE-L-1536 weight | 0.6560 | 0.6036 | 0.4531 |
| Selected low-weight setting | **0.6562** | **0.6050** | **0.4532** |

## Benchmark Context

The final 0.6604 score was obtained in a post-challenge target-domain
transductive setting. On the same evaluation server, it exceeds the reported
2025 Track 4 scores of UIT-OpenCubee (0.6493) and UT-T1 (0.6413), but it is
not an official award-eligible challenge result.

The machine-readable main progression is stored in
`results/official_metrics.csv`.
