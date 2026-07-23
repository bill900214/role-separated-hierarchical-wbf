# Transformer Prediction Inputs

These files are the Transformer-side prediction inputs retained for the reported Level III WBF.

## DEIMv2-DINOv3-S-832

```text
deimv2_832_conf005_iou070_official.json
```

- Resolution: 832 × 832
- Inference confidence: 0.05
- Inference IoU: 0.70
- Final Level III weight: 0.05

## DEIMv2-DINOv3-S-960

```text
submission_deimv2_960_conf052_iou045.json
```

- Resolution: 960 × 960
- Inference confidence: 0.52
- Inference IoU: 0.45
- Final Level III weight: 0.065
- Detections: 30,460
- Images represented: 1,000
- SHA-256: `2dc9f60a3c943a3710ddf790baae6e5380a70859512db8c0db97aea0310002bf`
- Release status: prediction-only; the original checkpoint is not redistributed

## D-FINE-L-1536

```text
dfine1536_last_conf005_iou070_official.json
```

- Resolution: 1536 × 1536
- Inference confidence: 0.05
- Inference IoU: 0.70
- Final Level III weight: 0.05

All files contain prediction results only. They do not contain hidden ground-truth annotations.
