# Legacy Experimental Scripts

This directory preserves original or historical experiment scripts for provenance.

- `wbf_fuse_results_original_found.py`  
  Retained original script using `mmdet.models.utils.weighted_boxes_fusion`. It contains the historical Day/Night threshold setting and environment-specific assumptions.

- `infer_YR_Y10_Y13_WBF_original.py`  
  Historical monolithic YOLO fusion script. Its default detector thresholds and single-scale structure do not represent the final multi-scale MSDNL pipeline.

These scripts are not the recommended public entry points. Refer to `scripts/`, `fusion/`, and `docs/` for the documented release workflow.
