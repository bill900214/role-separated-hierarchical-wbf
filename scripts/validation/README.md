# Validation Utilities

## Full Repository Audit

```bash
python scripts/validation/verify_repository.py
```

## Validate One Prediction JSON

```bash
python scripts/validation/validate_prediction_json.py \
  predictions/final/FINAL_MSDN_L_EC2.json \
  --expected-images 1000
```

## Compare Two Prediction JSON Files

```bash
python scripts/validation/compare_prediction_json.py expected.json actual.json
```

The comparison is order-independent and supports configurable decimal rounding.
