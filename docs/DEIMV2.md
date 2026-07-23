# DEIMv2 Auxiliary Predictions

The final fusion uses two DEIMv2-DINOv3-S prediction sources:

| Variant | Resolution | Epochs | Batch | WBF weight |
|---|---:|---:|---:|---:|
| DEIMv2-S-960 | 960 | 200 | 2 | 0.065 |
| DEIMv2-S-832 | 832 | 200 | 2 | 0.05 |

Reported shared settings:

```text
Optimizer: AdamW
Learning rate: 2e-4
Backbone learning rate: 1e-5
Weight decay: 1e-4
AMP: enabled
EMA: enabled
```

The repository may provide the final prediction JSON files without redistributing the entire upstream DEIMv2 framework. It must still document the upstream repository, commit hash, config, checkpoint, inference command, and checksum.

The exact DEIMv2-S-960 checkpoint remains to be verified from the original storage.
