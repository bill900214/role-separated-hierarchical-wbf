# Third-Party Components and File Provenance

| Component | Upstream repository | License | Local role |
|---|---|---|---|
| UT-T1 Track-4 release | public checkpoint source used by the study; exact revision not preserved | verify source and checkpoint terms | YOLO checkpoint provenance |
| YOLOR | https://github.com/WongKinYiu/yolor | GPL-3.0 | YOLOR-D6 inference integration |
| YOLOv10 | https://github.com/THU-MIG/yolov10 | AGPL-3.0 | multi-scale inference integration |
| YOLOv13 | https://github.com/iMoonLab/yolov13 | AGPL-3.0 | multi-scale inference integration |
| DEIMv2 | https://github.com/Intellindust-AI-Lab/DEIMv2 | Apache-2.0 | fisheye configs and DEIMv2-S-832 inference wrapper |
| D-FINE | https://github.com/Peterande/D-FINE | Apache-2.0 | three-stage fisheye configs and D-FINE-L-1536 inference wrapper |

## Repository File Map

- `code/inference/yolo/`: experiment-specific YOLO inference wrappers.
- `code/inference/deimv2/`: retained DEIMv2-S-832 wrapper and format utilities.
- `code/inference/dfine/`: retained D-FINE-L-1536 wrapper.
- `configs/transformer/deimv2/`: minimal retained DEIMv2 dependency chain.
- `configs/transformer/dfine/`: minimal retained D-FINE dependency chain.
- `legacy/`: original experimental scripts retained for provenance.

Exact upstream commit hashes were not stored in the collected experimental
archive. This limitation is disclosed rather than replaced with invented
revision identifiers.
