import os, json, glob, argparse
from PIL import Image
from collections import defaultdict

sceneList = ["M", "A", "E", "N"]

def get_image_id_from_name(path):
    name = os.path.basename(path).replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
    parts = name.split("_")
    cameraIdx = int(parts[0].replace("camera", ""))
    sceneIdx = sceneList.index(parts[1])
    frameIdx = int(parts[2])
    return int(str(cameraIdx) + str(sceneIdx) + str(frameIdx))

def load_image_sizes(img_dir):
    sizes = {}
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    for p in files:
        img_id = get_image_id_from_name(p)
        im = Image.open(p)
        sizes[img_id] = im.size
    return sizes

def clip(v):
    return max(0.0, min(1.0, v))

def xywh_to_xyxy_norm(b, w, h):
    x, y, bw, bh = b
    return [clip(x/w), clip(y/h), clip((x+bw)/w), clip((y+bh)/h)]

def xyxy_norm_to_xywh(b, w, h):
    x1, y1, x2, y2 = b
    x1, y1, x2, y2 = clip(x1)*w, clip(y1)*h, clip(x2)*w, clip(y2)*h
    x, y = min(x1, x2), min(y1, y2)
    bw, bh = abs(x2-x1), abs(y2-y1)
    return [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)]

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=area_a+area_b-inter
    return inter/union if union>0 else 0.0

def weighted_box(cluster):
    total = sum(x["score"] * x["weight"] for x in cluster)
    if total <= 0:
        return cluster[0]["box"]
    out = [0,0,0,0]
    for x in cluster:
        sw = x["score"] * x["weight"]
        for i in range(4):
            out[i] += x["box"][i] * sw
    return [v / total for v in out]

def fuse_group(items, iou_thr, total_weight):
    items = sorted(items, key=lambda x: x["score"] * x["weight"], reverse=True)
    clusters = []
    for item in items:
        best_i, best_iou = -1, 0
        for i, c in enumerate(clusters):
            cur = iou(item["box"], weighted_box(c))
            if cur > best_iou:
                best_iou, best_i = cur, i
        if best_i >= 0 and best_iou >= iou_thr:
            clusters[best_i].append(item)
        else:
            clusters.append([item])

    fused = []
    for c in clusters:
        box = weighted_box(c)
        score = sum(x["score"] * x["weight"] for x in c) / total_weight
        score = max(0.0, min(1.0, score))
        fused.append((box, score))
    return fused

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--jsons", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--iou_thr", type=float, default=0.60)
    ap.add_argument("--skip_box_thr", type=float, default=0.001)
    ap.add_argument("--final_thr", type=float, default=0.35)
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert len(args.jsons) == len(args.weights)
    sizes = load_image_sizes(args.img_dir)
    total_weight = sum(args.weights)

    grouped = defaultdict(list)

    for m, path in enumerate(args.jsons):
        data = json.load(open(path))
        weight = args.weights[m]
        for item in data:
            score = float(item["score"])
            if score < args.skip_box_thr:
                continue
            img_id = int(item["image_id"])
            cat = int(item["category_id"])
            if img_id not in sizes:
                continue
            w, h = sizes[img_id]
            grouped[(img_id, cat)].append({
                "box": xywh_to_xyxy_norm(item["bbox"], w, h),
                "score": score,
                "weight": weight
            })

    results = []
    for (img_id, cat), items in grouped.items():
        w, h = sizes[img_id]
        fused = fuse_group(items, args.iou_thr, total_weight)
        for box, score in fused:
            if score < args.final_thr:
                continue
            bbox = xyxy_norm_to_xywh(box, w, h)
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue
            results.append({
                "image_id": int(img_id),
                "category_id": int(cat),
                "bbox": bbox,
                "score": round(float(score), 6)
            })

    by_img = defaultdict(list)
    for r in results:
        by_img[r["image_id"]].append(r)

    final = []
    for img_id, arr in by_img.items():
        arr = sorted(arr, key=lambda x: x["score"], reverse=True)
        final.extend(arr[:args.topk])

    final = sorted(final, key=lambda x: (x["image_id"], x["category_id"], -x["score"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(final, open(args.out, "w"))

    print("saved:", args.out)
    print("detections:", len(final))
    print("unique images:", len(set(x["image_id"] for x in final)))
    print("first:", final[0] if final else "EMPTY")

if __name__ == "__main__":
    main()
