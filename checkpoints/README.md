# Checkpoint Availability

Checkpoint binaries are intentionally excluded from normal Git history.

## Retained Records

- D-FINE-L-1536 final checkpoint filename and SHA-256.
- D-FINE Stage-1 and Stage-2 selected checkpoint names.
- DEIMv2-DINOv3-S-832 checkpoint filename and SHA-256.
- DINOv3 distillation checkpoint filename and SHA-256.
- DEIMv2-S-960 prediction JSON provenance.

See:

```text
weight_inventory.txt
checkpoint_checksums_reference.txt
```

The reference checksum file documents hashes for binaries that are **not bundled** in this repository. It is not intended to be run with `sha256sum -c` against the current checkout.

## DEIMv2-S-960

The original DEIMv2-S-960 checkpoint was not retained in the collected experiment archive. The actual prediction JSON used by Level III is released instead:

```text
predictions/transformer_inputs/submission_deimv2_960_conf052_iou045.json
```

No missing checkpoint filename or checksum is invented.
