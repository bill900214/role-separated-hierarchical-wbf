#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a COCO-style object-detection prediction JSON."
    )
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--expected-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Prediction JSON must contain a list.")

    required = {"image_id", "category_id", "bbox", "score"}
    images = set()

    for index, item in enumerate(data):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Entry {index} is missing {sorted(missing)}.")

        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        bbox = item["bbox"]
        score = float(item["score"])

        if not 0 <= category_id <= 4:
            raise ValueError(
                f"Entry {index} has category_id={category_id}; expected 0-4."
            )
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Entry {index} has an invalid bbox.")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"Entry {index} has a non-positive bbox size.")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Entry {index} has score={score}; expected [0, 1].")

        images.add(image_id)

    if args.expected_images is not None and len(images) != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} represented images, found {len(images)}."
        )

    print(f"File: {args.json_file}")
    print(f"Detections: {len(data)}")
    print(f"Images represented: {len(images)}")
    print("Status: VALID")


if __name__ == "__main__":
    main()
