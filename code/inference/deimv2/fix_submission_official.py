import json

src = "submission_deimv2_infer1024_conf060_iou040.json"
dst = "submission_deimv2_infer1024_conf060_iou040_official.json"

sceneList = ["M", "A", "E", "N"]

data = json.load(open(src))
fixed = []

def get_image_id(name):
    name = str(name).split(".png")[0]
    parts = name.split("_")
    cameraIndx = int(parts[0].replace("camera", ""))
    sceneIndx = sceneList.index(parts[1])
    frameIndx = int(parts[2])
    return int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))

for item in data:
    new_item = dict(item)
    new_item["image_id"] = get_image_id(item["image_id"])
    new_item["category_id"] = int(item["category_id"])  # 保持 0~4
    fixed.append(new_item)

json.dump(fixed, open(dst, "w"))

print("saved:", dst)
print("detections:", len(fixed))
print("first:", fixed[0])
print("image_id type:", type(fixed[0]["image_id"]))
print("category min/max:", min(x["category_id"] for x in fixed), max(x["category_id"] for x in fixed))
print("unique images:", len(set(x["image_id"] for x in fixed)))
