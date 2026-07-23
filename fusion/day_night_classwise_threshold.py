#!/usr/bin/env python3
"""
Apply dataset-specific Day/Night Class-Wise Confidence Thresholding.

Night rule:
    str(image_id).startswith("293")

Class order / category_id:
    0: Bus
    1: Bike
    2: Car
    3: Pedestrian
    4: Truck

Final thresholds reported in the thesis:
    Day:   [0.28, 0.28, 0.28, 0.28, 0.23]
    Night: [0.08, 0.13, 0.18, 0.13, 0.18]

This script filters an existing COCO-style prediction JSON containing:
    image_id, category_id, bbox, score
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CLASS_NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]
DEFAULT_DAY_THRESHOLDS = [0.28, 0.28, 0.28, 0.28, 0.23]
DEFAULT_NIGHT_THRESHOLDS = [0.08, 0.13, 0.18, 0.13, 0.18]


def parse_thresholds(value: str) -> list[float]:
    """Parse a comma-separated five-value threshold vector."""
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "Threshold vector must contain exactly five values in the order "
            "Bus,Bike,Car,Pedestrian,Truck."
        )
    if any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("Every threshold must be between 0 and 1.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Day/Night class-wise thresholds to prediction JSON."
    )
    parser.add_argument("--src", type=Path, required=True, help="Input prediction JSON.")
    parser.add_argument("--dst", type=Path, required=True, help="Output filtered JSON.")
    parser.add_argument(
        "--night-prefix",
        default="293",
        help='image_id prefix treated as night. Default: "293".',
    )
    parser.add_argument(
        "--day-thresholds",
        type=parse_thresholds,
        default=DEFAULT_DAY_THRESHOLDS,
        help=(
            "Five comma-separated daytime thresholds in the order "
            "Bus,Bike,Car,Pedestrian,Truck. "
            "Default: 0.28,0.28,0.28,0.28,0.23"
        ),
    )
    parser.add_argument(
        "--night-thresholds",
        type=parse_thresholds,
        default=DEFAULT_NIGHT_THRESHOLDS,
        help=(
            "Five comma-separated nighttime thresholds in the order "
            "Bus,Bike,Car,Pedestrian,Truck. "
            "Default: 0.08,0.13,0.18,0.13,0.18"
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Maximum retained detections per image after filtering. Default: 300.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)

    if not isinstance(predictions, list):
        raise ValueError("Input JSON must be a list of prediction dictionaries.")
    return predictions


def is_night(image_id: Any, night_prefix: str) -> bool:
    return str(image_id).startswith(night_prefix)


def validate_prediction(prediction: dict[str, Any], index: int) -> None:
    required = {"image_id", "category_id", "bbox", "score"}
    missing = required.difference(prediction)
    if missing:
        raise ValueError(f"Prediction {index} is missing fields: {sorted(missing)}")

    category_id = int(prediction["category_id"])
    if not 0 <= category_id < len(CLASS_NAMES):
        raise ValueError(
            f"Prediction {index} has invalid category_id={category_id}; expected 0-4."
        )

    score = float(prediction["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Prediction {index} has invalid score={score}; expected [0, 1]."
        )

    bbox = prediction["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(
            f"Prediction {index} has invalid bbox; expected [x, y, width, height]."
        )


def apply_thresholds(
    predictions: list[dict[str, Any]],
    night_prefix: str,
    day_thresholds: list[float],
    night_thresholds: list[float],
    topk: int,
) -> list[dict[str, Any]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    day_kept = 0
    night_kept = 0

    for index, prediction in enumerate(predictions):
        validate_prediction(prediction, index)

        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        score = float(prediction["score"])

        night = is_night(image_id, night_prefix)
        thresholds = night_thresholds if night else day_thresholds

        if score >= thresholds[category_id]:
            by_image[image_id].append(prediction)
            if night:
                night_kept += 1
            else:
                day_kept += 1

    filtered: list[dict[str, Any]] = []
    for image_id in sorted(by_image):
        detections = sorted(
            by_image[image_id],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        filtered.extend(detections[:topk])

    print(f"Day detections retained: {day_kept}")
    print(f"Night detections retained: {night_kept}")
    print(f"Total retained after top-{topk}: {len(filtered)}")
    print(f"Unique images with detections: {len(by_image)}")
    return filtered


def main() -> None:
    args = parse_args()

    predictions = load_predictions(args.src)
    filtered = apply_thresholds(
        predictions=predictions,
        night_prefix=args.night_prefix,
        day_thresholds=args.day_thresholds,
        night_thresholds=args.night_thresholds,
        topk=args.topk,
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w", encoding="utf-8") as file:
        json.dump(filtered, file, ensure_ascii=False)

    print(f"Saved: {args.dst}")
    print(f"Night prefix: {args.night_prefix}")
    print(f"Class order: {CLASS_NAMES}")
    print(f"Day thresholds: {args.day_thresholds}")
    print(f"Night thresholds: {args.night_thresholds}")


if __name__ == "__main__":
    main()
