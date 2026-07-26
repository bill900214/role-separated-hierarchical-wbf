#!/usr/bin/env python3
"""
Portable WBF-like helper for prediction-artifact inspection.

IMPORTANT:
    The original reported Level-II and Level-III experiments used
    `mmdet.models.utils.weighted_boxes_fusion`.

    This file is a standalone approximation for Level-I inspection and
    implementation comparison. It is NOT claimed to reproduce the
    original MMDetection Level-II or Level-III outputs bit-for-bit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


SCENES = ["M", "A", "E", "N"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--jsons", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.65)
    parser.add_argument("--skip-box-thr", type=float, default=0.001)
    parser.add_argument("--final-thr", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def image_id_from_name(path: str | Path) -> int:
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) != 3 or not parts[0].startswith("camera"):
        raise ValueError(
            f"Unexpected image filename {Path(path).name!r}; expected "
            "camera<id>_<M|A|E|N>_<frame>.<ext>."
        )
    camera_index = int(parts[0].replace("camera", ""))
    scene_index = SCENES.index(parts[1])
    frame_index = int(parts[2])
    return int(f"{camera_index}{scene_index}{frame_index}")


def load_image_sizes(image_dir: Path) -> dict[int, tuple[int, int]]:
    sizes: dict[int, tuple[int, int]] = {}
    image_paths: list[str] = []
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(
            glob.glob(str(image_dir / "**" / extension), recursive=True)
        )

    for path in sorted(image_paths):
        with Image.open(path) as image:
            sizes[image_id_from_name(path)] = image.size

    if not sizes:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return sizes


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def xywh_to_normalized_xyxy(
    bbox: list[float], width: int, height: int
) -> list[float]:
    x, y, box_width, box_height = map(float, bbox)
    return [
        clip(x / width),
        clip(y / height),
        clip((x + box_width) / width),
        clip((y + box_height) / height),
    ]


def normalized_xyxy_to_xywh(
    box: list[float], width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = box
    x1, y1 = clip(x1) * width, clip(y1) * height
    x2, y2 = clip(x2) * width, clip(y2) * height
    x, y = min(x1, x2), min(y1, y2)
    box_width, box_height = abs(x2 - x1), abs(y2 - y1)
    return [
        round(x, 3),
        round(y, 3),
        round(box_width, 3),
        round(box_height, 3),
    ]


def intersection_over_union(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def weighted_box(cluster: list[dict[str, Any]]) -> list[float]:
    total = sum(item["score"] * item["weight"] for item in cluster)
    if total <= 0:
        return list(cluster[0]["box"])

    output = [0.0, 0.0, 0.0, 0.0]
    for item in cluster:
        score_weight = item["score"] * item["weight"]
        for index in range(4):
            output[index] += item["box"][index] * score_weight
    return [value / total for value in output]


def fuse_group(
    items: list[dict[str, Any]], iou_threshold: float, total_weight: float
) -> list[tuple[list[float], float]]:
    items = sorted(
        items,
        key=lambda item: item["score"] * item["weight"],
        reverse=True,
    )
    clusters: list[list[dict[str, Any]]] = []

    for item in items:
        best_index = -1
        best_iou = 0.0
        for index, cluster in enumerate(clusters):
            current_iou = intersection_over_union(item["box"], weighted_box(cluster))
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index

        if best_index >= 0 and best_iou >= iou_threshold:
            clusters[best_index].append(item)
        else:
            clusters.append([item])

    fused: list[tuple[list[float], float]] = []
    for cluster in clusters:
        score = (
            sum(item["score"] * item["weight"] for item in cluster)
            / total_weight
        )
        fused.append((weighted_box(cluster), clip(score)))
    return fused


def load_prediction_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a prediction list.")
    return data


def main() -> None:
    args = parse_args()
    if len(args.jsons) != len(args.weights):
        raise ValueError("--jsons and --weights must have equal length.")

    image_sizes = load_image_sizes(args.img_dir)
    total_weight = sum(args.weights)
    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for model_index, path in enumerate(args.jsons):
        model_weight = args.weights[model_index]
        for item in load_prediction_list(path):
            score = float(item["score"])
            if score < args.skip_box_thr:
                continue

            image_id = int(item["image_id"])
            category_id = int(item["category_id"])
            if image_id not in image_sizes:
                continue

            width, height = image_sizes[image_id]
            grouped[(image_id, category_id)].append(
                {
                    "box": xywh_to_normalized_xyxy(
                        item["bbox"], width, height
                    ),
                    "score": score,
                    "weight": model_weight,
                }
            )

    results: list[dict[str, Any]] = []
    for (image_id, category_id), items in grouped.items():
        width, height = image_sizes[image_id]
        for box, score in fuse_group(items, args.iou_thr, total_weight):
            if score < args.final_thr:
                continue

            bbox = normalized_xyxy_to_xywh(box, width, height)
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue

            results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": round(score, 6),
                }
            )

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_image[result["image_id"]].append(result)

    final: list[dict[str, Any]] = []
    for image_id in sorted(by_image):
        detections = sorted(
            by_image[image_id],
            key=lambda item: item["score"],
            reverse=True,
        )
        final.extend(detections[: args.topk])

    final.sort(
        key=lambda item: (
            item["image_id"],
            item["category_id"],
            -item["score"],
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        json.dump(final, file, ensure_ascii=False)

    print(f"Saved: {args.out}")
    print(f"Detections: {len(final)}")
    print(f"Unique images: {len({item['image_id'] for item in final})}")


if __name__ == "__main__":
    main()
