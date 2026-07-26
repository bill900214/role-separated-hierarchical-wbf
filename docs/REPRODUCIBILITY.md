# Reproducibility

## Directly Verifiable

The public release supports direct verification of:

- prediction JSON schemas and value ranges;
- SHA-256 checksums for retained prediction artifacts;
- the final Day/Night class-wise threshold stage;
- Level-I same-model multi-scale fusion from archived prediction inputs.

Run:

```bash
python scripts/validation/verify_repository.py
```

## Archived Prediction Inputs

The prediction JSON files required to inspect the reported fusion stages are
included under `predictions/`. They contain model outputs only and do not
contain hidden annotations.

## Level-II and Level-III WBF

The experiments used the MMDetection
`mmdet.models.utils.weighted_boxes_fusion` implementation. The exact
MMDetection/MMEngine environment versions were not retained. Therefore,
the committed Level-II and Level-III JSON files are the authoritative
experiment artifacts, and bit-for-bit recomputation across arbitrary
package versions is not claimed.

A standalone Level-I inspection helper is provided at:

```text
code/fusion/wbf_portable_helper.py
```

## Training Scope

D-FINE and DEIMv2 configuration records retained from the experiments are
included. Full training reproduction additionally requires the original
public datasets, pseudo-label source, upstream repositories, and model
initialization weights.

The DEIMv2-S-960 checkpoint was not retained; its actual prediction JSON
used in the final fusion is included.
