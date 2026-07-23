#!/usr/bin/env python3
"""Reference DEIMv2-DINOv3-S-832 inference wrapper.

Run this script with an upstream DEIMv2 checkout supplied through
`--deimv2-root`. It documents the retained 832 prediction branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import nms


SCENES = ["M", "A", "E", "N"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deimv2-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def image_id_from_path(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) != 3 or not parts[0].startswith("camera"):
        raise ValueError(
            f"Unexpected image filename {path.name!r}; expected "
            "camera<id>_<M|A|E|N>_<frame>.<ext>."
        )
    camera_index = int(parts[0].replace("camera", ""))
    scene_index = SCENES.index(parts[1])
    frame_index = int(parts[2])
    return int(f"{camera_index}{scene_index}{frame_index}")


def collect_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(image_dir.rglob(suffix))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return paths


def classwise_nms(
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kept_indices: list[int] = []
    for category in labels.unique():
        category_indices = torch.where(labels == category)[0]
        selected = nms(
            boxes[category_indices],
            scores[category_indices],
            iou_threshold,
        )
        kept_indices.extend(category_indices[selected].tolist())

    if not kept_indices:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4)),
            torch.empty((0,)),
        )

    keep = torch.tensor(kept_indices, dtype=torch.long)
    return labels[keep], boxes[keep], scores[keep]


def main() -> None:
    args = parse_args()

    deimv2_root = args.deimv2_root.resolve()
    if not deimv2_root.exists():
        raise FileNotFoundError(deimv2_root)
    sys.path.insert(0, str(deimv2_root))

    from engine.core import YAMLConfig  # type: ignore

    config = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = (
        checkpoint["ema"]["module"]
        if "ema" in checkpoint
        else checkpoint["model"]
    )
    config.model.load_state_dict(state)

    class DeploymentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = config.model.deploy()
            self.postprocessor = config.postprocessor.deploy()

        def forward(
            self,
            images: torch.Tensor,
            original_sizes: torch.Tensor,
        ):
            return self.postprocessor(self.model(images), original_sizes)

    model = DeploymentModel().to(args.device).eval()
    image_size = config.yaml_cfg["eval_spatial_size"]
    use_dinov3_normalization = bool(
        config.yaml_cfg.get("DINOv3STAs", False)
    )

    operations: list[object] = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    if use_dinov3_normalization:
        operations.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    preprocess = transforms.Compose(operations)

    results: list[dict[str, object]] = []
    image_paths = collect_images(args.image_dir)

    with torch.no_grad():
        for index, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            original_size = torch.tensor(
                [[width, height]],
                device=args.device,
            )
            image_tensor = preprocess(image).unsqueeze(0).to(args.device)

            labels, boxes, scores = model(image_tensor, original_size)
            labels = labels[0].detach().cpu()
            boxes = boxes[0].detach().cpu()
            scores = scores[0].detach().cpu()

            keep = scores >= args.conf
            labels, boxes, scores = (
                labels[keep],
                boxes[keep],
                scores[keep],
            )
            labels, boxes, scores = classwise_nms(
                labels,
                boxes,
                scores,
                args.iou,
            )

            image_id = image_id_from_path(image_path)
            for category, box, score in zip(labels, boxes, scores):
                x1, y1, x2, y2 = map(float, box.tolist())
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0 or box_height <= 0:
                    continue

                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category),
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(box_width, 2),
                            round(box_height, 2),
                        ],
                        "score": round(float(score), 6),
                    }
                )

            if index % 100 == 0:
                print(f"{index}/{len(image_paths)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)

    print(f"Saved: {args.output}")
    print(f"Detections: {len(results)}")


if __name__ == "__main__":
    main()
