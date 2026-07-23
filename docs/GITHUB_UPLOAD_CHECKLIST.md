# Public Release Checklist

Completed:

- [x] Remove private absolute server paths.
- [x] Remove credentials, tokens, private keys, and `.env` files.
- [x] Add the repository URL to `CITATION.cff`.
- [x] Document the Day/Night rule and final class-wise thresholds.
- [x] Add the final prediction JSON checksum.
- [x] Add the DEIMv2-S-960 prediction JSON checksum.
- [x] Document DEIMv2-S-960 as prediction-only.
- [x] Document retained D-FINE three-stage output directories.
- [x] Remove unused auxiliary Day/Night Transformer JSON files.

Remaining before claiming full end-to-end reproduction:

- [ ] Recover or explicitly select and validate the original MMDetection/MMEngine WBF environment.
- [ ] Verify exact Level-II and Level-III recomputation against the committed JSON artifacts.
- [ ] Add checkpoint download links only where redistribution is permitted.
- [ ] Preserve all required upstream license notices.
- [ ] Record exact upstream commit hashes from the original clones if they can be recovered.
- [ ] Add qualitative figures generated only from true prediction JSON files, when desired.
