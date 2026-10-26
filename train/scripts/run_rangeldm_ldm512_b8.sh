#!/usr/bin/env bash
# The original latent-prior run used 16/GPU at width 256.  Width 512 doubles
# the latent raster, so use the requested half batch (8/GPU).
set -euo pipefail

project="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec /opt/conda/envs/urbantwin-rangeldm/bin/torchrun \
  --standalone --nproc_per_node=8 \
  src/train_rangeldm_real_ldm.py \
  --data data/rangeldm_s2r_2k \
  --vae outputs/rangeldm_vae_m1_512_b4/epoch_020 \
  --output outputs/rangeldm_real_ldm_512_b8 \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --workers 4 \
  --width 512 \
  --save-every 20 \
  --seed 20260729
