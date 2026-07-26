#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EXPECTED_FINAL_COUNT = 33834
EXPECTED_FINAL_IMAGES = 1000
EXPECTED_FLOWCHART_SHA256 = "cb0399c5dfdf90cd547fdda930fa3ef3379a65fb0b5a75fc8764e13882127bfd"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list.")
    return data


def validate_prediction_file(path: Path) -> tuple[int, int]:
    required = {"image_id", "category_id", "bbox", "score"}
    images: set[int] = set()
    data = load_predictions(path)

    for index, item in enumerate(data):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"{path} entry {index} missing {sorted(missing)}")

        category_id = int(item["category_id"])
        bbox = item["bbox"]
        score = float(item["score"])

        if category_id not in range(5):
            raise ValueError(f"{path} entry {index} has invalid category.")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{path} entry {index} has invalid bbox.")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"{path} entry {index} has non-positive bbox.")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{path} entry {index} has invalid score.")

        images.add(int(item["image_id"]))

    return len(data), len(images)


def canonical(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        tuple(round(float(value), 6) for value in item["bbox"]),
        round(float(item["score"]), 6),
    )


def compare_json(expected: Path, actual: Path) -> None:
    left = Counter(canonical(x) for x in load_predictions(expected))
    right = Counter(canonical(x) for x in load_predictions(actual))
    if left != right:
        raise ValueError(
            f"Prediction mismatch: missing={sum((left-right).values())}, "
            f"extra={sum((right-left).values())}"
        )



def check_manuscript_alignment(root: Path) -> None:
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    if citation["title"] != 'Role-Separated Hierarchical Fusion for Multi-Scale YOLO–Transformer Fisheye Object Detection':
        raise ValueError("CITATION title does not match the manuscript.")
    expected = [("Tsai", "Chun-Ming"), ("Huang", "Ding-Jun"), ("Hsieh", "Jun-Wei"), ("Chang", "Ming-Ching")]
    actual = [(x["family-names"], x["given-names"]) for x in citation["authors"]]
    if actual != expected:
        raise ValueError(f"CITATION author order mismatch: {actual}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if 'Role-Separated Hierarchical Fusion for Multi-Scale YOLO–Transformer Fisheye Object Detection' not in readme or 'Chun-Ming Tsai, Ding-Jun Huang, Jun-Wei Hsieh, and Ming-Ching Chang' not in readme:
        raise ValueError("README title/authors do not match the manuscript.")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(f"Repository: {root}")

    python_files = list(root.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Python syntax: PASS ({len(python_files)} files)")

    shell_files = list(root.rglob("*.sh"))
    if shutil.which("bash"):
        for path in shell_files:
            result = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise ValueError(f"Shell syntax error in {path}")
        print(f"Shell syntax: PASS ({len(shell_files)} files)")

    yaml_files = list(root.rglob("*.yml")) + list(root.rglob("*.yaml"))
    for path in yaml_files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "__include__" in data:
            includes = data["__include__"]
            if isinstance(includes, str):
                includes = [includes]
            for include in includes:
                if not (path.parent / include).resolve().exists():
                    raise FileNotFoundError(f"Missing include: {path} -> {include}")
    print(f"YAML check: PASS ({len(yaml_files)} files)")

    prediction_files = list((root / "predictions").rglob("*.json"))
    stats = {path: validate_prediction_file(path) for path in prediction_files}
    print(f"Prediction schema: PASS ({len(prediction_files)} files)")

    checksum_file = root / "results/artifact_checksums.sha256"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split(maxsplit=1)
        path = root / rel.strip()
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
    print("Artifact checksums: PASS")

    final_json = root / "predictions/final/FINAL_MSDN_L_EC2.json"
    final_count, final_images = stats[final_json]
    if (final_count, final_images) != (
        EXPECTED_FINAL_COUNT,
        EXPECTED_FINAL_IMAGES,
    ):
        raise ValueError("Unexpected final prediction statistics.")
    print(f"Final prediction: {final_count} detections / {final_images} images")

    config = yaml.safe_load(
        (root / "configs/fusion/final_msdnl.yaml").read_text(encoding="utf-8")
    )
    assert config["class_order"] == ["Bus", "Bike", "Car", "Pedestrian", "Truck"]
    assert config["day_night_threshold"]["night_prefix"] == "293"
    assert config["day_night_threshold"]["day"] == [0.28, 0.28, 0.28, 0.28, 0.23]
    assert config["day_night_threshold"]["night"] == [0.08, 0.13, 0.18, 0.13, 0.18]
    assert config["level3"]["weights"] == [1.2, 0.065, 0.05, 0.05]
    assert config["level3"]["final_confidence_threshold"] == 0.295
    assert config["level3"]["top_k_per_image"] == 300
    print("Final configuration: PASS")

    rows = list(csv.DictReader(
        (root / "results/official_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        )
    ))
    if len(rows) != 6 or rows[-1]["f1"] != "0.6604":
        raise ValueError("Official result table mismatch.")
    print("Result table: PASS")

    check_manuscript_alignment(root)
    print("Manuscript title and authors: PASS")

    flowchart = root / "docs/assets/role_separated_hierarchical_wbf_pipeline.png"
    if digest(flowchart) != EXPECTED_FLOWCHART_SHA256:
        raise ValueError("Flowchart checksum mismatch.")
    print("Flowchart: PASS")

    source = root / "predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json"
    expected = root / "predictions/intermediate/MSDN_L_SEC_MS3_DN.json"
    script = root / "fusion/day_night_classwise_threshold.py"

    with tempfile.TemporaryDirectory() as directory:
        actual = Path(directory) / "daynight.json"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--src", str(source),
                "--dst", str(actual),
                "--night-prefix", "293",
                "--day-thresholds", "0.28,0.28,0.28,0.28,0.23",
                "--night-thresholds", "0.08,0.13,0.18,0.13,0.18",
                "--topk", "300",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        compare_json(expected, actual)

    print("Day/Night reconstruction: MATCH")
    print("Repository status: PASS")


if __name__ == "__main__":
    main()
