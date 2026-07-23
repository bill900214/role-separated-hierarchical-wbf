import os
import json
import cv2
import argparse
from tqdm import tqdm

CATEGORIES = [
    {"id": 0, "name": "bus"},
    {"id": 1, "name": "bike"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "pedestrian"},
    {"id": 4, "name": "truck"},
]

def collect_images(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
            files.append(name)
    files.sort()
    return files

def yolo_line_to_coco_bbox(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    cls_id = int(parts[0])
    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    bw = float(parts[3]) * img_w
    bh = float(parts[4]) * img_h

    left = x_center - bw / 2.0
    top = y_center - bh / 2.0

    return cls_id, [left, top, bw, bh]

def convert_sources_to_coco(sources, output_json):
    images = []
    annotations = []
    annotation_id = 0
    image_id = 0

    for source in sources:
        images_dir = source["images_dir"]
        labels_dir = source["labels_dir"]
        prefix = source.get("prefix", "")

        image_files = collect_images(images_dir)
        print(f"[INFO] Processing {images_dir}, total images = {len(image_files)}")

        for image_file in tqdm(image_files):
            image_path = os.path.join(images_dir, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"[WARN] Cannot read image: {image_path}")
                continue

            img_h, img_w = img.shape[:2]
            stem = os.path.splitext(image_file)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")

            coco_file_name = f"{prefix}{image_file}"

            images.append({
                "id": image_id,
                "file_name": coco_file_name,
                "width": img_w,
                "height": img_h
            })

            if not os.path.exists(label_path):
                image_id += 1
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parsed = yolo_line_to_coco_bbox(line, img_w, img_h)
                if parsed is None:
                    continue

                cls_id, bbox = parsed
                left, top, bw, bh = bbox

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [left, top, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "segmentation": []
                })
                annotation_id += 1

            image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"[INFO] Saved COCO json to: {output_json}")
    print(f"[INFO] images = {len(images)}, annotations = {len(annotations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources_json", type=str, required=True,
                        help="Path to a json file describing all YOLO sources")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output COCO json path")
    args = parser.parse_args()

    with open(args.sources_json, "r") as f:
        sources = json.load(f)

    convert_sources_to_coco(sources, args.output_json)
