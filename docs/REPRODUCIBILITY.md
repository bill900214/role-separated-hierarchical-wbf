# Reproducibility Scope

## Directly Verifiable

The repository supports direct verification of:

- prediction JSON schemas and score/bounding-box validity;
- SHA-256 checksums for the released prediction artifacts;
- the final Day/Night class-wise threshold stage;
- Level-I same-model multi-scale fusion from archived prediction inputs.

Run:

```bash
python scripts/validation/verify_repository.py
```

## Level-II and Level-III WBF

The experiments used `mmdet.models.utils.weighted_boxes_fusion`. The exact
MMDetection/MMEngine environment versions were not preserved. The committed
Level-II and Level-III JSON files are therefore the authoritative experiment
artifacts; bit-for-bit recomputation across arbitrary package versions is not
claimed.

A standalone helper for Level-I inspection is provided at
`code/fusion/wbf_portable_helper.py`.

## Training Scope

The public YOLO checkpoints are used for inference only. D-FINE and
DEIMv2-S-832 configuration records are included, but checkpoint binaries and
the training datasets are not redistributed. The DEIMv2-S-960 checkpoint was
not available in the final archive; its actual prediction JSON used in Level
III is included.
