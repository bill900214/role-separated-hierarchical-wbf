# Licensing and Reuse

## Repository-Level Position

The repository contains:

1. original experiment-specific utilities and documentation;
2. adapted inference/configuration files associated with upstream projects;
3. generated prediction JSON artifacts;
4. checkpoint provenance records without bundled checkpoint binaries.

One repository-wide license is intentionally not asserted over all files because upstream-derived materials remain governed by their corresponding source terms.

## Practical Reuse Guidance

- Review `third_party/README.md` before reusing detector-specific code or configurations.
- Preserve copyright, attribution, notice, and source-disclosure obligations required by the applicable upstream license.
- Do not assume that a model checkpoint has the same redistribution terms as its source code.
- Prediction JSON files contain generated model outputs and no hidden ground-truth annotations.
- Consult institutional or legal guidance when redistribution rights are uncertain.

## Why There Is No Root `LICENSE` File

A generic root license could incorrectly imply that every upstream-derived file is relicensed under one term. Instead, this repository provides explicit third-party notices and file provenance.

The absence of a root `LICENSE` file is deliberate and replaces the previous unfinished `LICENSE_SELECTION_REQUIRED.md` page.
