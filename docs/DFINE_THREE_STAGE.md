# D-FINE-L Three-Stage Fine-Tuning

D-FINE-L with HGNetv2-B4 is progressively fine-tuned at three resolutions.

| Paper stage | Fine-tuning source | Resolution | Epochs | Batch | Retained output directory |
|---|---|---:|---:|---:|---|
| Stage 1 | Official D-FINE-L pretrained checkpoint | 1280 × 1280 | 60 | 1 | `dfine_l_fisheye_1280_stage1_60e` |
| Stage 2 | Stage-1 `checkpoint0047.pth` | 1440 × 1440 | 60 | 1 | `dfine_l_fisheye_1440_v1_stage1_60e` |
| Stage 3 | Stage-2 epoch-42 checkpoint | 1536 × 1536 | 60 | 1 | `dfine_l_fisheye_1536_v1_from1440e42_60e` |

The original 1440 output directory retains the historical name `stage1_60e`. In the paper pipeline, that run corresponds to **Stage 2** of the progressive 1280 → 1440 → 1536 procedure.

The final Level III WBF uses the final converged Stage-3 `last.pth` checkpoint for D-FINE-L-1536.

## Retained Intermediate Checkpoint Records

```text
Stage 1 source for 1440:
dfine1280_checkpoint0047_F1_06063.pth

Stage 2 source for 1536:
dfine1440_e42_F1_06103.pth
```

These intermediate checkpoint names are documented for provenance. The repository does not claim that every checkpoint is redistributed.

## Reported Optimizer Settings

```text
Optimizer: AdamW
Learning rate: 2.5e-4
Backbone learning rate: 1.25e-5
Weight decay: 1.25e-4
AMP: enabled
EMA: enabled
```

## Configuration Files

```text
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1280.yml
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1440_v1.yml
configs/transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1536_v1.yml
```

## Reference Command Wrappers

```text
scripts/training/train_dfine_stage1_1280.sh
scripts/training/train_dfine_stage2_1440.sh
scripts/training/train_dfine_stage3_1536.sh
```

These are **reference wrappers reconstructed from the retained configurations, checkpoint records, and documented stage transitions**. They are not presented as verbatim shell-history records from the original experiments.

Before execution, set the upstream D-FINE repository, dataset paths, and checkpoint paths using the environment variables documented inside each wrapper.

Failed or nonfinal exploratory output directories, including runs marked `bad_ap`, `nan`, or the unused 1600-resolution run, are intentionally excluded from the public research workflow because they were not used by the final method.
