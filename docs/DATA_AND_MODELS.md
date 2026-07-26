# Data and Model Availability

## Publicly Released Here

- experiment-specific source code;
- inference and post-processing wrappers;
- final fusion configuration;
- retained D-FINE and DEIMv2 configuration files;
- prediction JSON files used in the reported fusion pipeline;
- official evaluation metrics and checksums.

## Not Redistributed

The following items are not included because they are external assets,
restricted evaluation material, or unavailable checkpoint binaries:

- FishEye8K and FishEye1K_eval images;
- hidden FishEye1K_eval annotations;
- VisDrone images and annotations;
- the public pseudo-label package used during Transformer fine-tuning;
- public UT-T1 checkpoint binaries;
- D-FINE and DEIMv2 checkpoint binaries.

Users should obtain datasets and upstream model weights from their original
providers and comply with the corresponding licenses and terms.

## Prediction Artifacts

Prediction JSON files contain only:

```text
image_id
category_id
bbox = [x, y, width, height]
score
```

They do not contain server-side ground truth.
