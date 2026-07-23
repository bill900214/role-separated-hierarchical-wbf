# Evaluation Protocol

Final evaluation uses the 1,000-image FishEye1K_eval set. The reference annotations remain on the official server.

Predictions use a COCO-style list of dictionaries with:

```text
image_id
category_id
bbox = [x, y, width, height]
score
```

The official server returned:

```text
F1-score
AP50–95
AP50
AP_S
AP_M
AP_L
```

No local script can reproduce hidden-test metrics without the server-side annotations. Local utilities should therefore validate format, category range, bounding-box validity, score range, image coverage, detection count, and checksums.
