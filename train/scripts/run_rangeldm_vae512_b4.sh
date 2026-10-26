#!/usr/bin/env bash
# High-resolution RangeLDM VAE: batch is intentionally halved from 8 to 4
# per GPU for the 512-column range raster on 32-GB V100 GPUs.
set -euo pipefail

project="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec /opt/conda/envs/urbantwin-rangeldm/bin/torchrun \
  --standalone --nproc_per_node=8 \
  src/train_rangeldm_vae.py \
  --data data/rangeldm_s2r_2k \
  --output outputs/rangeldm_vae_m1_512_b4 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --workers 4 \
  --width 512 \
  --save-every 5 \
  --levels 4 \
  --object-weight 8 \
  --boundary-weight 4 \
  --seed 20260729
