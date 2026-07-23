# DEIMv2 Auxiliary Prediction Branches

The final Level III WBF uses two DEIMv2-DINOv3-S prediction sources.

| Branch | Resolution | Epochs reported in experiment records | Batch | Inference confidence | Inference IoU | Level III weight | Release scope |
|---|---:|---:|---:|---:|---:|---:|---|
| DEIMv2-S-832 | 832 × 832 | 200 | 2 | 0.05 | 0.70 | 0.05 | Config, checkpoint record, prediction JSON |
| DEIMv2-S-960 | 960 × 960 | 200 | 2 | 0.52 | 0.45 | 0.065 | Prediction JSON only |

Reported shared settings:

```text
Optimizer: AdamW
Learning rate: 2e-4
Backbone learning rate: 1e-5
Weight decay: 1e-4
AMP: enabled
EMA: enabled
```

## Released Prediction Files

```text
predictions/transformer_inputs/deimv2_832_conf005_iou070_official.json
predictions/transformer_inputs/submission_deimv2_960_conf052_iou045.json
```

## DEIMv2-S-960 Release Scope

The actual DEIMv2-S-960 prediction input used by the reported final Level III WBF is released as:

```text
predictions/transformer_inputs/submission_deimv2_960_conf052_iou045.json
```

Artifact summary:

```text
Detections: 30,460
Images represented: 1,000
Minimum score: 0.520047
Maximum score: 0.985890
SHA-256: 2dc9f60a3c943a3710ddf790baae6e5380a70859512db8c0db97aea0310002bf
```

The original DEIMv2-S-960 checkpoint and its exact training-output directory were not retained in the collected experiment archive. Consequently:

- the actual prediction JSON is released;
- the branch remains part of the documented final fusion;
- checkpoint-level reproduction is not claimed;
- no checkpoint filename or checksum is invented.

The full upstream DEIMv2 framework is not duplicated in this repository. Users should consult the official upstream implementation and the experiment-specific files retained here.

## Prediction Format

Each prediction entry follows COCO-style output formatting:

```text
image_id
category_id
bbox = [x, y, width, height]
score
```

The JSON contains model predictions only and does not contain hidden ground-truth annotations.
