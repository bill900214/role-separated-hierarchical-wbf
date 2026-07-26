# Data and Model Availability

## Experimental Data

The Transformer fine-tuning pool contains 17,629 images derived from:

- the 5,288-image FishEye8K training split;
- class-mapped VisDrone samples;
- the public FishEye1K_eval pseudo-label package described in the paper.

The 2,712-image FishEye8K validation split is used for training monitoring and
checkpoint selection. No manual annotation of challenge test imagery was
performed, and server-side ground-truth annotations were not accessed.

## Included in This Repository

- project-specific source code;
- inference and post-processing wrappers;
- fusion and threshold configuration files;
- D-FINE and DEIMv2 configuration files available from the experiments;
- prediction JSON files used by the reported fusion pipeline;
- evaluation-server metrics and checksums.

## Not Redistributed

- FishEye8K, FishEye1K_eval, or VisDrone images and annotations;
- hidden FishEye1K_eval annotations;
- pseudo-label packages;
- public UT-T1 checkpoint binaries;
- D-FINE and DEIMv2 checkpoint binaries.

Datasets and upstream weights must be obtained from their original providers
under the corresponding terms.

## Prediction Artifacts

Prediction JSON files contain `image_id`, `category_id`, `bbox`, and `score`.
They do not contain hidden ground-truth annotations.
