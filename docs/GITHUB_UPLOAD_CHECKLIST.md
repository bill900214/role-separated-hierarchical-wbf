# Public Release Checklist

Completed:

- [x] Remove private absolute server paths and credentials.
- [x] Add the repository URL to `CITATION.cff`.
- [x] Document the Day/Night rule and final thresholds.
- [x] Add final and DEIMv2-S-960 JSON checksums.
- [x] Document DEIMv2-S-960 as prediction-only.
- [x] Document retained D-FINE three-stage output directories.
- [x] Remove unused auxiliary prediction JSON files.
- [x] Remove duplicate conversion utilities.
- [x] Remove unused Transformer configurations.
- [x] Separate historical scripts from recommended entry points.

Known release limitations:

- [ ] Original MMDetection/MMEngine WBF versions were not recovered.
- [ ] Exact Level-II and Level-III bit-for-bit recomputation is not claimed.
- [ ] Checkpoint binaries are not bundled.
- [ ] Exact upstream commit hashes were not recovered.
- [ ] Upstream license texts must be preserved when redistributing upstream-derived files.

These limitations are disclosed in the repository and must not be represented as completed.
