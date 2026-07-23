import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.datasets import letterbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv10-X inference for FishEye1K_eval")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--yolov10_model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Exact output JSON path")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--img_size", type=int, default=1280, choices=[1280, 1536])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--yolov10_conf", type=float, default=0.50)
    parser.add_argument("--yolov10_iou", type=float, default=0.65)
    return parser.parse_args()


def get_image_id(img_name: str) -> int:
    stem = Path(img_name).stem
    camera, scene, frame = stem.split("_")
    scene_list = ["M", "A", "E", "N"]
    camera_idx = int(camera.replace("camera", ""))
    scene_idx = scene_list.index(scene)
    frame_idx = int(frame)
    return int(f"{camera_idx}{scene_idx}{frame_idx}")


def preprocess_image(image: np.ndarray, img_size: int):
    img, ratio, pad = letterbox(image, new_shape=img_size, auto=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return img, ratio, pad


def scale_coords(coords, img0_shape, ratio, pad):
    coords = np.asarray(coords, dtype=np.float32)
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad[0]) / ratio[0]
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad[1]) / ratio[1]
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img0_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img0_shape[0])
    return coords.round().tolist()


def postprocess_yolov10(results, img_shape, conf_thres, ratio, pad):
    boxes, scores, classes = [], [], []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if float(conf) >= conf_thres:
                boxes.append(box.cpu().numpy())
                scores.append(float(conf.cpu()))
                classes.append(int(cls.cpu()))
    if boxes:
        boxes = scale_coords(boxes, img_shape, ratio, pad)
    return boxes, scores, classes


def main() -> None:
    args = parse_args()
    model = YOLO(args.yolov10_model)

    image_files = sorted(
        os.path.join(root, filename)
        for root, _, files in os.walk(args.image_folder)
        for filename in files
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        raise FileNotFoundError(f"No evaluation images found under {args.image_folder}")

    submission = []
    elapsed_ms = 0.0

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img_shape = image.shape[:2]
        image_id = get_image_id(os.path.basename(img_path))
        processed, ratio, pad = preprocess_image(image, args.img_size)
        tensor = torch.from_numpy(processed).unsqueeze(0).to(args.device)

        start = time.perf_counter()
        results = model.predict(
            tensor,
            imgsz=args.img_size,
            conf=args.yolov10_conf,
            iou=args.yolov10_iou,
            device=args.device,
            verbose=False,
        )
        elapsed_ms += (time.perf_counter() - start) * 1000.0

        boxes, scores, classes = postprocess_yolov10(
            results, img_shape, args.yolov10_conf, ratio, pad
        )
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            submission.append(
                {
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                }
            )

    fps = 1000.0 * len(image_files) / elapsed_ms
    print(f"Processed {len(image_files)} images in {elapsed_ms / 1000.0:.2f}s")
    print(f"FPS: {fps:.2f}; normalized FPS: {min(fps, 25.0) / 25.0:.4f}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.output_dir) / (
            f"infer_Y10_yolov10x_{args.img_size}_"
            f"c{args.yolov10_conf:.3f}_i{args.yolov10_iou:.3f}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(submission, file, ensure_ascii=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
