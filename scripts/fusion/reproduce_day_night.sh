#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository}"

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
OUTPUT="${1:-$PROJECT_ROOT/predictions/intermediate/MSDN_L_SEC_MS3_DN_reproduced.json}"

python "$PROJECT_ROOT/fusion/day_night_classwise_threshold.py" \
  --src "$PROJECT_ROOT/predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json" \
  --dst "$OUTPUT" \
  --night-prefix 293 \
  --day-thresholds 0.28,0.28,0.28,0.28,0.23 \
  --night-thresholds 0.08,0.13,0.18,0.13,0.18 \
  --topk 300

python "$PROJECT_ROOT/scripts/validation/compare_prediction_json.py" \
  "$PROJECT_ROOT/predictions/intermediate/MSDN_L_SEC_MS3_DN.json" \
  "$OUTPUT"
