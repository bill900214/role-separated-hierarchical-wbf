import os
import sys
import json
from glob import glob

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig


# =========================
# 你只要改這幾個地方
# =========================

CONFIG = "configs/deimv2/deimv2_dinov3_s_fisheye_infer1024.yml"
WEIGHT = "outputs/deimv2_dinov3_s_fisheye_batch2_832_amp_200e/best_stg2.pth"
IMG_DIR = "/data/datasets/fisheye_test/images"

OUT_JSON = "submission_deimv2_infer1024_conf060_iou040.json"

CONF_THRES = 0.60
IOU_THRES = 0.40

DEVICE = "cuda"


def load_model():
    cfg = YAMLConfig(CONFIG, resume=WEIGHT)

    checkpoint = torch.load(WEIGHT, map_location="cpu")

    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    # 重要：
    # 832 訓練權重直接換 1024 / 1280 inference config 會出現 anchor mismatch
    # 所以這裡跳過 decoder.anchors 和 decoder.valid_mask
    skip_keys = [
        "decoder.anchors",
        "decoder.valid_mask",
    ]

    new_state = {}
    for k, v in state.items():
        if any(sk in k for sk in skip_keys):
            print("Skip:", k)
            continue
        new_state[k] = v

    missing, unexpected = cfg.model.load_state_dict(new_state, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(DEVICE).eval()
    return model, cfg


@torch.no_grad()
def main():
    model, cfg = load_model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get("DINOv3STAs", False)

    print("Inference size:", img_size)
    print("CONFIG:", CONFIG)
    print("WEIGHT:", WEIGHT)
    print("CONF:", CONF_THRES)
    print("IOU:", IOU_THRES)

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) if vit_backbone else T.Lambda(lambda x: x)
    ])

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths += glob(os.path.join(IMG_DIR, ext))

    image_paths = sorted(image_paths)

    print("Total images:", len(image_paths))

    results = []

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        orig_size = torch.tensor([[w, h]], device=DEVICE)
        img_tensor = transforms(img).unsqueeze(0).to(DEVICE)

        labels, boxes, scores = model(img_tensor, orig_size)

        labels = labels[0].detach().cpu()
        boxes = boxes[0].detach().cpu()
        scores = scores[0].detach().cpu()

        keep = scores >= CONF_THRES
        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]

        final_keep = []

        if len(labels) > 0:
            for cls in labels.unique():
                cls_idx = torch.where(labels == cls)[0]
                keep_idx = nms(boxes[cls_idx], scores[cls_idx], IOU_THRES)
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

        image_id = os.path.splitext(os.path.basename(img_path))[0]

        for cls, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box.tolist()

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            results.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(bw), 2),
                    round(float(bh), 2)
                ],
                "score": round(float(score), 6)
            })

        if idx % 100 == 0:
            print(f"{idx}/{len(image_paths)}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f)

    print("Saved:", OUT_JSON)
    print("Detections:", len(results))


if __name__ == "__main__":
    main()
