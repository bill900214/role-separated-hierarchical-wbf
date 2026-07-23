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

## Configuration files

```text
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1280.yml
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1440_v1.yml
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1536_v1.yml
```

## Command wrappers

The repository provides:

```text
scripts/training/train_dfine_stage1_1280.sh
scripts/training/train_dfine_stage2_1440.sh
scripts/training/train_dfine_stage3_1536.sh
```

The wrappers use the upstream D-FINE tuning interface (`-t checkpoint.pth`). Before execution, set the checkpoint and repository paths in the environment variables shown inside each script.

The wrappers reproduce the documented stage transitions. GPU count, master port, and seed can be adjusted to match the local environment.
