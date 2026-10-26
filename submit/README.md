# submit - Final submission reproduction

Reproduces the final loopy-account submission
(`submission_final_repro.zip` via `run_final_repro.sh`):
quantile_axis synthetic frames (Realism 0.7905) +
VoxelRCNN predictions (Detection 0.5583, 10170 boxes),
Combined 0.6512.

```bash
cd loopy_code/submit
bash run_final_repro.sh
```

Contains: quantile_axis generation code, VoxelRCNN checkpoint (identical to the
submitted model), official test frames, inference/packaging code, all relative paths.
Details: `docs/FINAL_SUBMISSION_REPRODUCTION.md`.
