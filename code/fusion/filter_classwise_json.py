import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--thr", required=True, help="格式：0:0.44,1:0.38,2:0.42,3:0.34,4:0.44")
parser.add_argument("--topk", type=int, default=300)
args = parser.parse_args()

thr = {}
for p in args.thr.split(","):
    k, v = p.split(":")
    thr[int(k)] = float(v)

data = json.load(open(args.src))
kept = []

for x in data:
    c = int(x["category_id"])
    s = float(x["score"])
    if s >= thr.get(c, 0.40):
        kept.append(x)

by_img = defaultdict(list)
for x in kept:
    by_img[int(x["image_id"])].append(x)

final = []
for img_id, arr in by_img.items():
    arr = sorted(arr, key=lambda x: x["score"], reverse=True)
    final.extend(arr[:args.topk])

json.dump(final, open(args.dst, "w"))

print("saved:", args.dst)
print("detections:", len(final))
print("unique images:", len(set(x["image_id"] for x in final)))
print("thr:", thr)
