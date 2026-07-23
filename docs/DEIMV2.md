# DEIMv2 Auxiliary Predictions

The final fusion uses two DEIMv2-DINOv3-S prediction sources:

| Variant | Resolution | Epochs | Batch | Inference conf. | Inference IoU | WBF weight |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-S-960 | 960 | 200 | 2 | 0.52 | 0.45 | 0.065 |
| DEIMv2-S-832 | 832 | 200 | 2 | 0.05 | 0.70 | 0.05 |

Reported shared settings:

```text
Optimizer: AdamW
Learning rate: 2e-4
Backbone learning rate: 1e-5
Weight decay: 1e-4
AMP: enabled
EMA: enabled
```

## Released prediction files

```text
predictions/transformer_inputs/deimv2_832_conf005_iou070_official.json
predictions/transformer_inputs/submission_deimv2_960_conf052_iou045.json
```

The full upstream DEIMv2 framework is not duplicated in this repository. Users should clone the official upstream implementation and apply the provided fisheye configuration and inference utilities.

The DEIMv2-S-832 checkpoint is documented in `checkpoints/weight_inventory.txt`. The original DEIMv2-S-960 checkpoint filename was not preserved in the collected experiment archive. The exact prediction JSON used by the reported final fusion is released, but this repository does not claim to redistribute the original 960 checkpoint.
