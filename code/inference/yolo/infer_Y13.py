import argparse
import json
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


SCENE_LIST = ["M", "A", "E", "N"]


def get_image_id(img_name: str) -> int:
    """Convert camera_scene_frame filename to the official integer image_id."""
    stem = Path(img_name).stem
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Unexpected image filename: {img_name!r}. "
            "Expected camera<id>_<M|A|E|N>_<frame>.<ext>."
        )

    camera_idx = int(parts[0].replace("camera", ""))
    scene_idx = SCENE_LIST.index(parts[1])
    frame_idx = int(parts[2])
    return int(f"{camera_idx}{scene_idx}{frame_idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv13-L inference for FishEye1K_eval."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the YOLOv13-L PyTorch checkpoint.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the evaluation image directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output COCO-style prediction JSON.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1280,
        choices=[1280, 1536],
        help="Inference resolution. The final method uses 1280 and 1536.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.495,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model = YOLO(args.model)

    image_paths = sorted(
        os.path.join(directory, filename)
        for directory, _, filenames in os.walk(args.image_dir)
        for filename in filenames
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No evaluation images found under {args.image_dir}")

    results_json: list[dict[str, object]] = []
    total_process_time = 0.0

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        start_time = time.perf_counter()
        prediction = model.predict(
            source=image,
            imgsz=args.img_size,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        total_process_time += time.perf_counter() - start_time

        boxes = prediction.boxes.cpu().numpy()
        for box, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = map(float, box)
            results_json.append(
                {
                    "image_id": get_image_id(img_path),
                    "category_id": int(class_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

    fps = len(image_paths) / total_process_time
    normalized_fps = min(fps, 25.0) / 25.0
    print(f"Processed {len(image_paths)} images in {total_process_time:.2f}s")
    print(f"FPS: {fps:.2f}, normalized FPS: {normalized_fps:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results_json, file, ensure_ascii=False)

    print(f"Saved submission JSON to {output_path}")


if __name__ == "__main__":
    main(parse_args())
