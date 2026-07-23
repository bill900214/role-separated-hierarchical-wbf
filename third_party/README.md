# Third-Party Components and File Provenance

This project relies on upstream implementations. Their licenses remain applicable to upstream-derived files and dependencies.

| Component | Upstream repository | Reported license | Local role |
|---|---|---|---|
| UT-T1 Track-4 release | public checkpoint source used by the study; exact revision not preserved | review source terms before redistribution | YOLO checkpoint provenance |
| YOLOR | https://github.com/WongKinYiu/yolor | GPL-3.0 | YOLOR-D6 inference integration |
| YOLOv10 | https://github.com/THU-MIG/yolov10 | AGPL-3.0 | multi-scale inference integration |
| YOLOv13 | https://github.com/iMoonLab/yolov13 | AGPL-3.0 | multi-scale inference integration |
| DEIMv2 | https://github.com/Intellindust-AI-Lab/DEIMv2 | verify the exact upstream revision | fisheye configs and DEIMv2-S-832 inference wrapper |
| D-FINE | https://github.com/Peterande/D-FINE | Apache-2.0 | three-stage fisheye configs and D-FINE-L-1536 inference wrapper |

## Repository File Map

- `code/inference/yolo/`: experiment-specific YOLO inference wrappers.
- `code/inference/deimv2/`: retained DEIMv2-S-832 wrapper and format utilities.
- `code/inference/dfine/`: retained D-FINE-L-1536 wrapper.
- `configs/transformer/deimv2/`: minimal retained DEIMv2 configuration dependency chain.
- `configs/transformer/dfine/`: minimal retained D-FINE configuration dependency chain.
- `legacy/`: original experimental scripts retained for provenance, not recommended as portable entry points.

Exact upstream commit hashes were not stored in the collected experiment archive. This repository identifies the projects and discloses that limitation rather than inventing revision identifiers.

See `THIRD_PARTY_NOTICES.md` and `docs/LICENSING_AND_REUSE.md`.
