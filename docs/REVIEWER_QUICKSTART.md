# Reviewer Quick Start

This page provides a short verification path that does not require hidden annotations or model checkpoints.

## 1. Install Portable Dependencies

```bash
python -m pip install -r requirements.txt
```

## 2. Run the Repository Audit

```bash
python scripts/validation/verify_repository.py
```

The audit checks:

- Python syntax;
- shell syntax where Bash is available;
- YAML parsing and include targets;
- all prediction JSON schemas;
- category IDs, scores, and bounding-box dimensions;
- artifact SHA-256 files;
- final configuration constants;
- absence of private server paths;
- exact Day/Night reconstruction.

Expected summary:

```text
Final prediction: 33834 detections / 1000 images
DEIMv2-S-960: 30460 detections / 1000 images
Day/Night reconstruction: MATCH
Repository status: PASS
```

## 3. Inspect the Method-to-Artifact Mapping

See:

```text
docs/METHOD_ARTIFACT_MAP.md
configs/fusion/final_msdnl.yaml
```

## 4. Optional Level-I Reconstruction

FishEye1K_eval image files are required to read image dimensions:

```bash
export PROJECT_ROOT=/path/to/role-separated-hierarchical-wbf
export EVAL_IMAGES=/path/to/FishEye1K_eval/images
bash scripts/fusion/reproduce_level1.sh
```

## Evaluation Boundary

Hidden-test AP and F1 metrics cannot be recalculated locally because the reference annotations remain on the official evaluation platform. The repository provides the submitted prediction JSON and the platform-returned metric record.
