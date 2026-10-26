# Original Training (train directory): reproduce the detection model from scratch

The detection model = VoxelRCNN (GBlobs fork), trained on
RangeLDM redrawn batch B (2k) + d50k (50k points/frame) + pedestrian alignment v3 recipe.
Below is the full chain that regenerates the training data from raw sources and trains
a model identical to the bundled checkpoint, using only the code in this directory.

## Dependency chain

```
raw UrbanTwin digital twin ──▶ base corpus 10k + sequence-level split (G0S)
raw V2X-Real public M1 ──────▶ RangeLDM-S2R redrawn batch B (2k)
                                    │
                                    └─▶ d50k (50k points/frame, 2k)
                                            │
                                            └─▶ pedestrian alignment v3 (training-time recipe)
                                                    │
                                                    ▼
                              VoxelRCNN training (4 GPUs, batch 16, 40 epochs) -> checkpoint
```

## 1. Environment and raw data

- conda: `urbantwin-v2xreal` (top-level `environment.yml`);
  RangeLDM training additionally uses the `urbantwin-rangeldm` environment
  (code in `third_party/RangeLDM`).
- Raw data (download from official sources; not shipped):
  - UrbanTwin digital-twin parts -> `data/urbantwin_parts/`
  - V2X-Real public training LiDAR -> `data/v2xreal_public/`

## 2. Base corpus and sequence-level split (S1)

```bash
bash scripts/01_extract_and_prepare.sh
python src/build_sequence_splits_v2.py ...      # contiguous 250-frame sequence split (anti-leakage)
```

## 3. RangeLDM redrawn batch B (S2)

```bash
python src/build_rangeldm_s2r_dataset.py ...    # 128x1024 range-image pairs (real M1 + projected UrbanTwin)
bash scripts/run_rangeldm_vae512_b4.sh          # VAE training
bash scripts/run_rangeldm_ldm512_b8.sh          # latent diffusion model training
python src/infer_rangeldm_sdedit.py --batch B   # controlled redraw, 2k frames (batches A/B; final uses B)
```

## 4. d50k density-stabilized source (S3)

Resample every frame of the redrawn batch B to exactly 50,000 points (float32 XYZ):

```bash
python src/build_detector_rangeldm_dataset.py --points-per-frame 50000 ...
# equivalent NumPy: rng.choice(pts, 50000, replace=len(pts)<50000) per frame
```

## 5. Pedestrian alignment v3 (S4, training-time recipe)

The augmentor lives in
`third_party/GBlobs/pcdet/datasets/augmentor/data_augmentor.py` (`v2x_ped_align`):
pedestrians rescaled to [0.68,0.69,1.74] m + SPARSE_PROB 0.6 (1-2 azimuth wedges + 20-45% random points).
The training config in `configs/` enables it.

## 6. VoxelRCNN training

```bash
cd loopy_code/train
conda run -n urbantwin-v2xreal python third_party/GBlobs/tools/train.py \
  --launcher pytorch --cfg_file configs/voxel_rcnn_gblobs_rangeldm2k_b_d50k_pedalign_v3.yaml \
  --batch_size 16 --epochs 40 --workers 8 --fix_random_seed \
  --extra_tag voxelrcnn_g0s_rangeldm2k_b_d50k_pedalign_v3_4gpu_b16
```
The produced checkpoint
(`third_party/GBlobs/output/.../<extra_tag>/ckpt/checkpoint_epoch_40.pth`) is the
same recipe as the checkpoint bundled under `submit/third_party/GBlobs/checkpoints/`.

Compliance: train with synthetic labels only; real frames for validation/model
selection only; frames in `forbidden_frames.txt` are always excluded.
