# Artifact Release Status

## Verified

- All retained prediction JSON files pass schema and value validation.
- Final prediction and DEIMv2-S-960 checksums match.
- Day/Night thresholding exactly reconstructs the committed relaxed-threshold JSON.
- Level-I YOLOv13-L reconstruction is verified.
- Level-I YOLOv10-X reconstruction is verified with numerical tolerance near 1e-6.
- Private absolute server paths and credential patterns are absent.

## Archived with Explicit Limitation

- Level II and Level III outputs are provided as authoritative experiment artifacts.
- The exact original MMDetection/MMEngine WBF versions were not retained.
- The portable WBF helper is not described as a bit-for-bit replacement for the original Level II/III environment.
- D-FINE training wrappers are reconstructed reference wrappers, not verbatim shell-history records.

## Not Bundled

- FishEye1K_eval images and hidden annotations.
- Detector checkpoint binaries.
- Original DEIMv2-S-960 checkpoint.
- Exact upstream commit hashes that were not preserved in the collected archive.

## Public-Release Prerequisite

Before public redistribution, review all upstream source and checkpoint terms listed in `THIRD_PARTY_NOTICES.md`.
