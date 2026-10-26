# loopy_code - final submission reproduction (Combined 0.6512)

Two subdirectories:

- **[submit/](submit/)** - submission reproduction. `bash submit/run_final_repro.sh`
  generates the quantile_axis synthetic frames, runs VoxelRCNN inference (10170 boxes),
  and packages the final submission bundle (launch script, checkpoint, official test frames).
- **[train/](train/)** - original training. Data generation code (UrbanTwin base / G0S,
  RangeLDM redraw, d50k, pedestrian alignment v3) + GBlobs training code +
  official RangeLDM code + training config, reproducing the bundled checkpoint from scratch.

Environment: conda `urbantwin-v2xreal` (`environment.yml`, py3.10 + torch2.1.2 cu118 + spconv-cu118).
All paths are relative to each directory. Derived/raw data are not shipped.
