# Final Submission Reproduction (submit directory)

Submitted bundle: the final loopy-account submission (2026-08-15, Final phase).
Official result: **Combined 0.6512** = 0.6 x 0.5583 (Detection) + 0.4 x 0.7905 (Realism)
- Detection: mAP@0.5 0.3908 / BEV mAP@0.5 0.4219 / mAP@0.7 0.1047;
  per-class vehicle 0.5672 / pedestrian 0.0825 / truck 0.5227;
  10170 predictions (vehicle 4133 / pedestrian 3752 / truck 2285) on 1547 ground-truth boxes.
- Realism: CD 6.90 / MMD 1.25e-2 / EMD 2.26 / FPD 9.34.

## One-click run

```bash
cd loopy_code/submit
bash run_final_repro.sh
# optional: DEVICE=1 bash run_final_repro.sh
#           PREDICTIONS=my_preds.json bash run_final_repro.sh   # reuse existing predictions
```

## 1) Data: quantile_axis (Realism 0.7905)

- Script: `src/generate_realism_baselines.py` (candidate `quantile_axis`)
- Method: for each of x/y/z axes independently, build a piecewise-linear mapping from
  2001 quantiles (`linspace(0,1,2001)`) of the source distribution to the target
  distribution, aligning each marginal; then seeded resampling to 50 x 2000 points,
  clipped to [-40,-40,-8,40,40,2].
- Inputs: `data/base_synthetic/` (50 base synthetic frames),
  `data/real_ref_frames/` (real reference frames, layout `<scenario>/-1/*.bin`)
- Output: 50 frames x 2000 points, float32 XYZ (24,000 bytes/frame, 100,000 points total)
- Local proxy 0.9210 (72 frames) -> official 0.7905: per-axis mapping breaks 3D geometry
  (CD/FPD increase sharply), which is why the final system kept the geometry-preserving
  affine calibration instead.

## 2) Detection: VoxelRCNN (Detection 0.5583)

- Model: two-stage Voxel R-CNN (GBlobs fork), trained on
  RangeLDM redrawn batch B + d50k + pedestrian alignment v3 recipe,
  4 GPUs, global batch 16, 40 epochs, fixed random seed
  (checkpoint bundled and identical to the submitted one).
- Inference: direct forward pass on the official 50 test frames
  (**no test-time augmentation**; flip-TTA was evaluated and dropped),
  GBLOBS=1.
- Output: 10170 boxes (vehicle 4133 / pedestrian 3752 / truck 2285).

## 3) Packaging and verification

- `src/build_submission.py` assembles `predictions.json + synthetic/ + declaration.txt`.
- Verification: box count 10170. For per-box comparison, place the archived
  `predictions_voxrcnn_new_test.json` under `outputs/` and compare manually.

## Data prerequisites (not shipped)

| Path | Content |
|---|---|
| `data/base_synthetic/` | 50 base synthetic frames (selected from the UrbanTwin corpus; see ../train/docs/TRAINING_0TO1.md) |
| `data/real_ref_frames/` | public real reference frames, layout `<scenario>/-1/*.bin` (e.g., V2X-Real public training frames) |
