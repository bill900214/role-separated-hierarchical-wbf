# Evaluation Protocol

Final evaluation uses the 1,000-image FishEye1K_eval set. The reference annotations remain on the official evaluation platform.

Predictions use COCO-style dictionaries with:

```text
image_id
category_id
bbox = [x, y, width, height]
score
```

The platform returned:

```text
F1-score
AP50–95
AP50
AP_S
AP_M
AP_L
```

No local script can reproduce hidden-test metrics without the server-side annotations.

## Local Integrity Validation

```bash
python scripts/validation/validate_prediction_json.py \
  predictions/final/FINAL_MSDN_L_EC2.json \
  --expected-images 1000
```

Artifact hashes are listed in:

```text
results/artifact_checksums.sha256
```
