import os
import sys
import json
import argparse
from glob import glob

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def load_model(config, resume, device):
    cfg = YAMLConfig(config, resume=resume)

    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device).eval()
    return model, cfg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--img_dir", required=True)
    parser.add_argument("-o", "--out_json", default="submission.json")
    parser.add_argument("--txt_dir", default="infer_txt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("-d", "--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.txt_dir, exist_ok=True)

    model, cfg = load_model(args.config, args.resume, args.device)

    img_size = cfg.yaml_cfg["eval_spatial_size"]

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])

    results = []

    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths += glob(os.path.join(args.img_dir, ext))

    img_paths = sorted(img_paths)
    print("Total images:", len(img_paths))

    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        orig_size = torch.tensor([[w, h]], device=args.device)
        img_tensor = transform(img).unsqueeze(0).to(args.device)

        labels, boxes, scores = model(img_tensor, orig_size)

        labels = labels[0].detach().cpu()
        boxes = boxes[0].detach().cpu()
        scores = scores[0].detach().cpu()

        keep = scores >= args.conf
        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]

        final_keep = []
        for cls in labels.unique():
            cls_idx = torch.where(labels == cls)[0]
            cls_boxes = boxes[cls_idx]
            cls_scores = scores[cls_idx]
            keep_idx = nms(cls_boxes, cls_scores, args.iou)
            final_keep.extend(cls_idx[keep_idx].tolist())

        if len(final_keep) > 0:
            final_keep = torch.tensor(final_keep)
            labels = labels[final_keep]
            boxes = boxes[final_keep]
            scores = scores[final_keep]
        else:
            labels = torch.empty((0,), dtype=torch.long)
            boxes = torch.empty((0, 4))
            scores = torch.empty((0,))

        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(args.txt_dir, stem + ".txt")

        with open(txt_path, "w") as f:
            for cls, box, score in zip(labels, boxes, scores):
                x1, y1, x2, y2 = box.tolist()
                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    continue

                cls_id = int(cls)

                results.append({
                    "image_id": stem,
                    "category_id": cls_id,
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(bw), 2),
                        round(float(bh), 2)
                    ],
                    "score": round(float(score), 6)
                })

                f.write(
                    f"{cls_id} {x1:.2f} {y1:.2f} {bw:.2f} {bh:.2f} {float(score):.6f}\n"
                )

        if idx % 100 == 0:
            print(f"{idx}/{len(img_paths)}")

    with open(args.out_json, "w") as f:
        json.dump(results, f)

    print("Saved:", args.out_json)
    print("Detections:", len(results))
    print("TXT dir:", args.txt_dir)


if __name__ == "__main__":
    main()
