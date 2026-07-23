import argparse
import json
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from models.experimental import attempt_load as load_yolor
from utils.datasets import letterbox
from utils.general import non_max_suppression

warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*indexing argument.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOR-D6 inference for FishEye1K_eval")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--yolor_model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Exact output JSON path")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--img_size", type=int, default=1280)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--yolor_conf", type=float, default=0.05)
    parser.add_argument("--yolor_iou", type=float, default=0.70)
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
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad[0]) / ratio[0]
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad[1]) / ratio[1]
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img0_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img0_shape[0])
    return coords.round()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_yolor(args.yolor_model).to(device)
    model.eval()

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
        tensor = torch.from_numpy(processed).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            prediction = model(tensor)[0]
            prediction = non_max_suppression(
                prediction,
                conf_thres=args.yolor_conf,
                iou_thres=args.yolor_iou,
            )
        elapsed_ms += (time.perf_counter() - start) * 1000.0

        for detections in prediction:
            if detections is None or not len(detections):
                continue
            detections[:, :4] = torch.tensor(
                scale_coords(detections[:, :4].cpu().numpy(), img_shape, ratio, pad),
                device=detections.device,
                dtype=detections.dtype,
            )
            for *xyxy, confidence, class_id in detections:
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                submission.append(
                    {
                        "image_id": image_id,
                        "category_id": int(class_id.cpu()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(confidence.cpu()),
                    }
                )

    fps = 1000.0 * len(image_files) / elapsed_ms
    print(f"Processed {len(image_files)} images in {elapsed_ms / 1000.0:.2f}s")
    print(f"FPS: {fps:.2f}; normalized FPS: {min(fps, 25.0) / 25.0:.4f}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.output_dir) / (
            f"infer_YR_yolor_d6_{args.img_size}_"
            f"c{args.yolor_conf:.3f}_i{args.yolor_iou:.3f}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(submission, file, ensure_ascii=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
