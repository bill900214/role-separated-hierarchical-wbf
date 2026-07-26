# Setup

## Portable Validation Environment

```bash
python -m pip install -r requirements.txt
python scripts/validation/verify_repository.py
```

## Detector Environments

YOLOR, YOLOv10, YOLOv13, D-FINE, and DEIMv2 should be installed from
their official upstream repositories.

The inference wrappers use environment variables for upstream source roots,
checkpoint paths, and image directories. Example commands are provided in:

```text
scripts/inference/run_yolo_multiscale.sh
scripts/inference/run_transformer_inputs.sh
```

## Dataset Paths

Update dataset placeholders in a local copy of the retained D-FINE and
DEIMv2 configuration files. Do not commit local absolute paths or
credentials.
