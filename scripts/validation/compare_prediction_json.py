#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def canonical(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        tuple(round(float(value), 6) for value in item["bbox"]),
        round(float(item["score"]), 6),
    )


def load(path: Path) -> list[tuple[Any, ...]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list.")
    return sorted(canonical(item) for item in data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two prediction JSON files after canonical sorting."
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    expected = load(args.expected)
    actual = load(args.actual)

    print(f"Expected detections: {len(expected)}")
    print(f"Actual detections: {len(actual)}")

    if expected != actual:
        expected_set = set(expected)
        actual_set = set(actual)
        print(f"Missing canonical entries: {len(expected_set - actual_set)}")
        print(f"Extra canonical entries: {len(actual_set - expected_set)}")
        raise SystemExit(1)

    print("Status: MATCH")


if __name__ == "__main__":
    main()
