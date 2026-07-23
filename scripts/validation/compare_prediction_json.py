#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two prediction JSON files after canonical rounding."
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Decimal places used for bbox and score comparison.",
    )
    return parser.parse_args()


def canonical(
    item: dict[str, Any],
    decimals: int,
) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        tuple(round(float(value), decimals) for value in item["bbox"]),
        round(float(item["score"]), decimals),
    )


def load(path: Path, decimals: int) -> Counter[tuple[Any, ...]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list.")
    return Counter(canonical(item, decimals) for item in data)


def main() -> None:
    args = parse_args()
    expected = load(args.expected, args.decimals)
    actual = load(args.actual, args.decimals)

    print(f"Expected detections: {sum(expected.values())}")
    print(f"Actual detections: {sum(actual.values())}")

    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        print(f"Missing canonical entries: {sum(missing.values())}")
        print(f"Extra canonical entries: {sum(extra.values())}")
        raise SystemExit(1)

    print("Status: MATCH")


if __name__ == "__main__":
    main()
