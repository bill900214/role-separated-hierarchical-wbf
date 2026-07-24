#!/usr/bin/env python3
"""Run portable integrity and reproducibility checks for this repository."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
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
EXPECTED_DEIM960_COUNT = 30460
EXPECTED_DEIM960_IMAGES = 1000

PRIVATE_PATTERNS = (
    "/data/students/",
    "/data/datasets/",
    "BEGIN PRIVATE KEY",
    "ghp_",
    "github_pat_",
)


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
        raise ValueError(f"{path} must contain a prediction list.")
    return data


def validate_predictions(path: Path) -> tuple[int, int]:
    data = load_predictions(path)
    images: set[int] = set()
    required = {"image_id", "category_id", "bbox", "score"}

    for index, item in enumerate(data):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"{path}: entry {index} missing {sorted(missing)}")

        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        bbox = item["bbox"]
        score = float(item["score"])

        if category_id not in range(5):
            raise ValueError(f"{path}: invalid category_id={category_id}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{path}: invalid bbox at entry {index}")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"{path}: non-positive bbox at entry {index}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{path}: invalid score={score}")

        images.add(image_id)

    return len(data), len(images)


def verify_sidecar(path: Path, sidecar: Path) -> None:
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = digest(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, found {actual}"
        )


def canonical(item: dict[str, Any], decimals: int = 6) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        tuple(round(float(value), decimals) for value in item["bbox"]),
        round(float(item["score"]), decimals),
    )


def compare_json(expected: Path, actual: Path, decimals: int = 6) -> None:
    left = Counter(canonical(item, decimals) for item in load_predictions(expected))
    right = Counter(canonical(item, decimals) for item in load_predictions(actual))
    if left != right:
        missing = left - right
        extra = right - left
        raise ValueError(
            f"Prediction mismatch: missing={sum(missing.values())}, "
            f"extra={sum(extra.values())}"
        )


def check_yaml_includes(root: Path) -> int:
    checked = 0
    for path in root.rglob("*.yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        checked += 1
        if isinstance(data, dict) and "__include__" in data:
            includes = data["__include__"]
            if isinstance(includes, str):
                includes = [includes]
            for include in includes:
                target = (path.parent / include).resolve()
                if not target.exists():
                    raise FileNotFoundError(
                        f"Missing YAML include from {path}: {include}"
                    )
    return checked


def check_final_config(root: Path) -> None:
    config_path = root / "configs/fusion/final_msdnl.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["class_order"] == [
        "Bus",
        "Bike",
        "Car",
        "Pedestrian",
        "Truck",
    ]
    assert config["day_night_threshold"]["night_prefix"] == "293"
    assert config["day_night_threshold"]["day"] == [
        0.28,
        0.28,
        0.28,
        0.28,
        0.23,
    ]
    assert config["day_night_threshold"]["night"] == [
        0.08,
        0.13,
        0.18,
        0.13,
        0.18,
    ]
    assert config["level2"]["weights"] == [9.0, 9.0, 9.0]
    assert config["level2"]["normalized_weight_ratio"] == [1.0, 1.0, 1.0]
    assert config["level3"]["weights"] == [1.2, 0.065, 0.05, 0.05]
    assert config["level3"]["final_confidence_threshold"] == 0.295
    assert config["level3"]["top_k_per_image"] == 300




def check_official_metrics(root: Path) -> None:
    path = root / "results/official_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    expected = [
        {"method": "Historical Raw Equal-Weight WBF", "f1": "0.5719", "ap50_95": "0.6360", "ap50": "0.8739", "ap_s": "0.4983", "ap_m": "0.7426", "ap_l": "0.6268"},
        {"method": "Historical Day/Night Thresholding", "f1": "0.6377", "ap50_95": "0.6033", "ap50": "0.8080", "ap_s": "0.4531", "ap_m": "0.7284", "ap_l": "0.6203"},
        {"method": "Best Recorded YOLO-Only Stage (IoU=0.6575)", "f1": "0.6425", "ap50_95": "", "ap50": "", "ap_s": "", "ap_m": "", "ap_l": ""},
        {"method": "Selected YOLO-Transformer Setting", "f1": "0.6562", "ap50_95": "0.6050", "ap50": "0.8060", "ap_s": "0.4532", "ap_m": "0.7325", "ap_l": "0.6196"},
        {"method": "Same-Model Multi-Scale YOLO", "f1": "0.6596", "ap50_95": "0.6123", "ap50": "0.8170", "ap_s": "0.4665", "ap_m": "0.7362", "ap_l": "0.6214"},
        {"method": "Final Scene-Specific Thresholding (MSDNL)", "f1": "0.6604", "ap50_95": "0.6147", "ap50": "0.8220", "ap_s": "0.4709", "ap_m": "0.7378", "ap_l": "0.6214"},
    ]

    if len(rows) != len(expected):
        raise ValueError(
            f"Expected {len(expected)} official metric rows, found {len(rows)}."
        )

    keys = ("method", "f1", "ap50_95", "ap50", "ap_s", "ap_m", "ap_l")
    for actual, wanted in zip(rows, expected):
        for key in keys:
            if actual[key] != wanted[key]:
                raise ValueError(
                    f"Official metric mismatch for {wanted['method']} / {key}: "
                    f"expected {wanted[key]!r}, found {actual[key]!r}"
                )



def check_method_figure(root: Path) -> None:
    path = root / "docs/assets/role_separated_hierarchical_wbf_pipeline.png"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing method flowchart: {path}")
    expected = "cb0399c5dfdf90cd547fdda930fa3ef3379a65fb0b5a75fc8764e13882127bfd"
    actual = digest(path)
    if actual != expected:
        raise ValueError(
            f"The original manuscript flowchart was modified: {actual}"
        )


def check_release_assets(root: Path) -> None:
    required = [
        root / "LICENSE",
        root / "NOTICE",
        root / "THIRD_PARTY_NOTICES.md",
        root / "docs/assets/role_separated_hierarchical_wbf_pipeline.png",
        root / "docs/DATA_AND_ANNOTATION_STATUS.md",
        root / "docs/MANUSCRIPT_METADATA.md",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing release asset: {path}")


def check_manuscript_metadata(root: Path) -> None:
    citation = yaml.safe_load(
        (root / "CITATION.cff").read_text(encoding="utf-8")
    )
    expected_title = "Role-Separated Hierarchical Fusion for Multi-Scale YOLO–Transformer Fisheye Object Detection"
    if citation["title"] != expected_title:
        raise ValueError(
            f"CITATION title mismatch: {citation['title']!r}"
        )

    expected_authors = [
        ("Tsai", "Chun-Ming"),
        ("Huang", "Ding-Jun"),
        ("Hsieh", "Jun-Wei"),
        ("Chang", "Ming-Ching"),
    ]
    actual_authors = [
        (author["family-names"], author["given-names"])
        for author in citation["authors"]
    ]
    if actual_authors != expected_authors:
        raise ValueError(
            f"CITATION author order mismatch: {actual_authors}"
        )

    readme = (root / "README.md").read_text(encoding="utf-8")
    if expected_title not in readme:
        raise ValueError("README does not contain the exact manuscript title.")
    author_line = (
        "Chun-Ming Tsai, Ding-Jun Huang, Jun-Wei Hsieh, "
        "and Ming-Ching Chang"
    )
    if author_line not in readme:
        raise ValueError("README author order does not match the manuscript.")

def check_private_patterns(root: Path) -> None:
    skip_suffixes = {
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".zip",
        ".pt",
        ".pth",
        ".ckpt",
    }
    verifier_path = Path(__file__).resolve()
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() in skip_suffixes
            or path.resolve() == verifier_path
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PRIVATE_PATTERNS:
            if pattern in text:
                raise ValueError(f"Private/sensitive pattern {pattern!r} in {path}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(f"Repository: {root}")

    # Python syntax
    python_files = list(root.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Python syntax: PASS ({len(python_files)} files)")

    # Shell syntax when Bash is available
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
                raise ValueError(f"Shell syntax error in {path}: {result.stderr}")
        print(f"Shell syntax: PASS ({len(shell_files)} files)")
    else:
        print("Shell syntax: SKIPPED (bash unavailable)")

    yaml_count = check_yaml_includes(root)
    print(f"YAML parse/include check: PASS ({yaml_count} files)")

    prediction_files = list((root / "predictions").rglob("*.json"))
    stats: dict[Path, tuple[int, int]] = {}
    for path in prediction_files:
        stats[path] = validate_predictions(path)
    print(f"Prediction schema check: PASS ({len(prediction_files)} files)")

    final_json = root / "predictions/final/FINAL_MSDN_L_EC2.json"
    final_sha = root / "predictions/final/FINAL_MSDN_L_EC2.sha256"
    deim960_json = (
        root
        / "predictions/transformer_inputs"
        / "submission_deimv2_960_conf052_iou045.json"
    )
    deim960_sha = (
        root
        / "predictions/transformer_inputs"
        / "submission_deimv2_960_conf052_iou045.sha256"
    )

    verify_sidecar(final_json, final_sha)
    verify_sidecar(deim960_json, deim960_sha)
    print("Artifact sidecar checksums: PASS")

    final_count, final_images = stats[final_json]
    deim_count, deim_images = stats[deim960_json]

    if (final_count, final_images) != (
        EXPECTED_FINAL_COUNT,
        EXPECTED_FINAL_IMAGES,
    ):
        raise ValueError(
            f"Unexpected final prediction stats: {final_count}/{final_images}"
        )
    if (deim_count, deim_images) != (
        EXPECTED_DEIM960_COUNT,
        EXPECTED_DEIM960_IMAGES,
    ):
        raise ValueError(
            f"Unexpected DEIMv2-S-960 stats: {deim_count}/{deim_images}"
        )

    print(
        f"Final prediction: {final_count} detections / {final_images} images"
    )
    print(f"DEIMv2-S-960: {deim_count} detections / {deim_images} images")

    check_final_config(root)
    print("Final fusion configuration: PASS")

    check_official_metrics(root)
    print("Official result progression: PASS")

    check_method_figure(root)
    print("Method flowchart: PASS")

    check_release_assets(root)
    print("Release assets and license: PASS")

    check_manuscript_metadata(root)
    print("Manuscript title and authors: PASS")

    check_private_patterns(root)
    print("Private path/credential scan: PASS")

    # Exact Day/Night reconstruction
    source = root / "predictions/intermediate/SEC_MS3_Y10Y13MS_raw.json"
    expected = root / "predictions/intermediate/MSDN_L_SEC_MS3_DN.json"
    script = root / "fusion/day_night_classwise_threshold.py"

    with tempfile.TemporaryDirectory() as directory:
        actual = Path(directory) / "daynight.json"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--src",
                str(source),
                "--dst",
                str(actual),
                "--night-prefix",
                "293",
                "--day-thresholds",
                "0.28,0.28,0.28,0.28,0.23",
                "--night-thresholds",
                "0.08,0.13,0.18,0.13,0.18",
                "--topk",
                "300",
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
