#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "$0")/.." && pwd)"
data_dir="$project/data/rangeldm_s2r_2k"
log_dir="$project/outputs/rangeldm_vae_m1"
mkdir -p "$log_dir"

while true; do
  count=$(find "$data_dir/real" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "$count" -ge 2000 && -s "$data_dir/manifest.json" ]]; then
    break
  fi
  sleep 20
done

cd "$project"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec /opt/conda/envs/urbantwin-rangeldm/bin/torchrun \
  --standalone --nproc_per_node=8 \
  src/train_rangeldm_vae.py \
  --data data/rangeldm_s2r_2k \
  --output outputs/rangeldm_vae_m1 \
  --epochs 20 \
  --batch-size 8 \
  --workers 4 \
  --width 256 \
  --save-every 5
