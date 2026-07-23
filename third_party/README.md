# Third-Party Components

This project relies on upstream implementations. Their licenses remain applicable to upstream-derived files and checkpoints.

| Component | Upstream repository | License | Local use/modifications |
|---|---|---|---|
| UT-T1 Track-4 release | Public checkpoint source used by the study; exact upstream revision was not preserved in the collected archive | Review source terms before redistribution | YOLO checkpoint provenance, inference and format-conversion adaptations |
| YOLOR | https://github.com/WongKinYiu/yolor | GPL-3.0 | YOLOR-D6 inference wrapper and COCO-style JSON conversion |
| YOLOv10 | https://github.com/THU-MIG/yolov10 | AGPL-3.0 | Multi-scale inference wrapper and COCO-style JSON conversion |
| YOLOv13 | https://github.com/iMoonLab/yolov13 | AGPL-3.0 | Multi-scale inference wrapper and COCO-style JSON conversion |
| DEIMv2 | https://github.com/Intellindust-AI-Lab/DEIMv2 | Apache-2.0 | Fisheye configs, inference utilities, and prediction JSON files |
| D-FINE | https://github.com/Peterande/D-FINE | Apache-2.0 | Three-stage fisheye configs, command wrappers, and inference utilities |

## Revision note

Exact upstream commit hashes were not stored in the collected experiment archive. The repository therefore identifies upstream projects and preserves the experiment-specific files, but does not invent commit hashes. If the original server clones remain available, their commit hashes should be recorded in a later archival revision. No commit hash is invented in this release.

Do not apply a blanket license to upstream-derived files without preserving the corresponding upstream notices and license requirements.
