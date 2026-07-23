# D-FINE-L Three-Stage Fine-Tuning

D-FINE-L uses HGNetv2-B4 and is progressively fine-tuned at three resolutions.

| Stage | Fine-tuning source | Resolution | Epochs | Batch |
|---|---|---:|---:|---:|
| Stage 1 | official pretrained checkpoint | 1280 | 60 | 1 |
| Stage 2 | Stage-1 checkpoint0047.pth | 1440 | 60 | 1 |
| Stage 3 | Stage-2 epoch-42 checkpoint | 1536 | 60 | 1 |

The final Level III WBF uses the final converged Stage-3 `last.pth` checkpoint for D-FINE-L-1536.

Reported optimizer settings:

```text
Optimizer: AdamW
Learning rate: 2.5e-4
Backbone learning rate: 1.25e-5
Weight decay: 1.25e-4
AMP: enabled
EMA: enabled
```

The exact original shell commands must be recovered from shell history, logs, or training records before the public release.
