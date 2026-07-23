# D-FINE-L Three-Stage Fine-Tuning

D-FINE-L with HGNetv2-B4 is progressively fine-tuned at three resolutions.

| Paper stage | Fine-tuning source | Resolution | Epochs | Batch | Retained output directory |
|---|---|---:|---:|---:|---|
| Stage 1 | Official D-FINE-L pretrained checkpoint | 1280 × 1280 | 60 | 1 | `dfine_l_fisheye_1280_stage1_60e` |
| Stage 2 | Stage-1 `checkpoint0047.pth` | 1440 × 1440 | 60 | 1 | `dfine_l_fisheye_1440_v1_stage1_60e` |
| Stage 3 | Stage-2 epoch-42 checkpoint | 1536 × 1536 | 60 | 1 | `dfine_l_fisheye_1536_v1_from1440e42_60e` |

The historical 1440 directory name contains `stage1_60e`; within the paper pipeline it corresponds to Stage 2 of the 1280 → 1440 → 1536 sequence.

The final Level III WBF uses the final converged Stage-3 `last.pth` checkpoint.

## Retained Intermediate Checkpoint Names

```text
dfine1280_checkpoint0047_F1_06063.pth
dfine1440_e42_F1_06103.pth
```

## Reported Optimizer Settings

```text
Optimizer: AdamW
Learning rate: 2.5e-4
Backbone learning rate: 1.25e-5
Weight decay: 1.25e-4
AMP: enabled
EMA: enabled
```

## Configurations

```text
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1280.yml
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1440_v1.yml
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1536_v1.yml
```

## Reference Training Wrappers

```text
scripts/training/train_dfine_stage1_1280.sh
scripts/training/train_dfine_stage2_1440.sh
scripts/training/train_dfine_stage3_1536.sh
```

These wrappers were reconstructed from retained configurations, checkpoint records, and documented stage transitions. They are not represented as verbatim original shell-history records.

Failed or nonfinal exploratory runs marked `bad_ap`, `nan`, or resolution 1600 are intentionally excluded because they were not used by the final method.
