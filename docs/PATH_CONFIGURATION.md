# Path Configuration

Private server paths have been replaced with portable placeholders.

## Common Environment Variables

```text
PROJECT_ROOT
EVAL_IMAGES
YOLOR_ROOT
YOLOV10_ROOT
YOLOV13_ROOT
DEIMV2_ROOT
DFINE_ROOT
```

Checkpoint variables are documented inside each shell wrapper.

## Training Configuration Paths

The retained YAML files contain placeholders such as:

```text
/path/to/project
```

Replace them in a local copy or parameterize them through the upstream framework. Do not commit personal account paths, API tokens, private keys, or hidden evaluation annotations.
