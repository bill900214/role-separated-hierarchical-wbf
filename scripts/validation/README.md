# Validation

Run the complete public-release integrity check:

```bash
python scripts/validation/verify_repository.py
```

Validate one prediction JSON:

```bash
python scripts/validation/validate_prediction_json.py       predictions/final/FINAL_MSDN_L_EC2.json       --expected-images 1000
```
