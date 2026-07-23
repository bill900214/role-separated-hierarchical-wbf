import os
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms

from engine.core import YAMLConfig

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.001

transform = transforms.Compose([
    transforms.Resize((832, 832)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()

def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

def load_model(cfg_path, ckpt_path):
    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    model = cfg.model.to(DEVICE)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema" in ckpt:
        state = ckpt["ema"]["module"] if isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"] else ckpt["ema"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    return model, cfg

def postprocess_outputs(cfg, outputs, orig_h, orig_w):
    if hasattr(cfg, "postprocessor") and cfg.postprocessor is not None:
        target_sizes = torch.tensor([[orig_h, orig_w]], device=DEVICE)
        results = cfg.postprocessor(outputs, target_sizes)
        results = results[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"].detach().cpu().numpy()
        return boxes, scores, labels

    if isinstance(outputs, (list, tuple)):
        results = outputs[0]
    else:
        results = outputs

    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    labels = results["labels"].detach().cpu().numpy()
    return boxes, scores, labels

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg = load_model(args.config, args.resume)
    image_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.*")))

    print(f"[INFO] found {len(image_paths)} images")
    print(f"[INFO] ckpt = {args.resume}")
    print(f"[INFO] out_dir = {args.out_dir}")

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            if idx % 100 == 0:
                print(f"[INFO] processing {idx}/{len(image_paths)}")

            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(x)

            boxes, scores, labels = postprocess_outputs(cfg, outputs, orig_h, orig_w)

            txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            txt_path = os.path.join(args.out_dir, txt_name)

            with open(txt_path, "w") as f:
                for box, score, cls_id in zip(boxes, scores, labels):
                    if float(score) < CONF_THRES:
                        continue
                    cx, cy, bw, bh = xyxy_to_yolo(box, orig_w, orig_h)
                    f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(score):.6f}\n")

    print("[INFO] done")

if __name__ == "__main__":
    main()