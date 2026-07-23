# Experiment Configuration Files

Only the configuration dependency chains used by the reported experiments are retained.

## Fusion

```text
fusion/final_msdnl.yaml
```

## D-FINE-L Progressive Fine-Tuning

```text
transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1280.yml
transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1440_v1.yml
transformer/dfine/dfine/custom/objects365/dfine_hgnetv2_l_fisheye_1536_v1.yml
```

Their required dataset, runtime, dataloader, optimizer, and model include files are retained in the same relative structure.

## DEIMv2-DINOv3-S-832

```text
transformer/deimv2/deimv2/deimv2_dinov3_s_fisheye.yml
```

The DEIMv2-S-960 branch is released as prediction-only because the original checkpoint and exact 960 configuration were not retained in the collected archive.
