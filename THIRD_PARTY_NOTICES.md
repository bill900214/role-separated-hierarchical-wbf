# Third-Party Notices

This repository contains experiment-specific wrappers, configurations, and archival artifacts associated with several upstream projects.

| Component | Upstream project | Upstream license reported by the project | Repository use |
|---|---|---|---|
| YOLOR | https://github.com/WongKinYiu/yolor | GPL-3.0 | YOLOR-D6 inference integration |
| YOLOv10 | https://github.com/THU-MIG/yolov10 | AGPL-3.0 | multi-scale inference integration |
| YOLOv13 | https://github.com/iMoonLab/yolov13 | AGPL-3.0 | multi-scale inference integration |
| DEIMv2 | https://github.com/Intellindust-AI-Lab/DEIMv2 | review the exact upstream revision | fisheye configuration and inference records |
| D-FINE | https://github.com/Peterande/D-FINE | Apache-2.0 | three-stage fisheye configuration and inference records |
| UT-T1 Track-4 release | public checkpoint source used in the study | exact source terms were not preserved in the collected archive | YOLO checkpoint provenance |

No blanket relicensing of upstream-derived files is asserted by this repository.

Before redistributing an upstream-derived file or checkpoint, review the exact upstream revision and preserve all notices required by that source. Checkpoint binaries are not bundled in normal Git history.

Additional details are available in `docs/LICENSING_AND_REUSE.md` and `third_party/README.md`.
