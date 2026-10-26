#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
COND="$PROJECT/data/rangeldm_s2r_ped2k/synthetic"
GEN="$PROJECT/outputs/rangeldm_sdedit_ped2k_anchor"
OUT="$PROJECT/data/openpcdet_g0s_seq250_rangeldm_bped4k"

while [ "$(find "$COND" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -lt 2000 ]; do
  sleep 30
done

mkdir -p "$GEN"
for i in 0 1 2 3; do
  start=$((i * 500))
  CUDA_VISIBLE_DEVICES=$((i + 4)) conda run -n urbantwin-rangeldm python "$PROJECT/src/infer_rangeldm_sdedit.py" \
    --data "$COND" --output "$GEN/shard_${i}" --strengths 0.05 --frames 500 --start "$start" \
    --batch-size 8 --anchor-objects > "$GEN/shard_${i}.log" 2>&1 &
done
wait

PROJECT_ROOT="$PROJECT" /opt/conda/envs/urbantwin-rangeldm/bin/python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
root = Path(os.environ["PROJECT_ROOT"]) / "outputs/rangeldm_sdedit_ped2k_anchor"
audits = sorted(root.glob("shard_*/strength_0.05/audit.json"))
rows = [row for path in audits for row in json.loads(path.read_text())["frames"]]
if len(rows) != 2000 or len({row["frame"] for row in rows}) != 2000:
    raise SystemExit(f"incomplete generation: {len(rows)} frames")
recall = np.array([row["object_ray_recall"] for row in rows], dtype=float)
depth = np.array([row["object_depth_mae_m"] for row in rows], dtype=float)
if recall.min() < 0.999 or depth.max() > 1e-6:
    raise SystemExit(f"anchor audit failed: recall_min={recall.min()} depth_max={depth.max()}")
print(json.dumps({"frames": len(rows), "object_recall_min": float(recall.min()), "object_depth_mae_max": float(depth.max())}))
PY

conda run -n urbantwin-v2xreal python "$PROJECT/src/build_detector_rangeldm_multiblock_dataset.py" \
  --base-root "$PROJECT/data/openpcdet_g0s_seq250" \
  --info-root "$PROJECT/data/openpcdet_g0s_seq250_rangeldm2k" \
  --block "rb:$PROJECT/data/openpcdet_g0s_seq250_rangeldm_block_b_source/ImageSets/train.txt:$PROJECT/outputs/rangeldm_sdedit_2k_anchor_b" \
  --block "rp:$PROJECT/data/openpcdet_g0s_seq250_rangeldm_ped_block_source_ids.txt:$GEN" \
  --output-root "$OUT"

cd "$PROJECT"
exec env CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n urbantwin-v2xreal torchrun --standalone --nproc_per_node=4 \
  third_party/OpenPCDet/tools/train.py --launcher pytorch --cfg_file configs/dsvt_vfield_g0s_rangeldm2k.yaml \
  --batch_size 32 --epochs 40 --workers 8 --extra_tag g0s_rangeldm_bped4k_scratch_4gpu_b32 --fix_random_seed \
  --set DATA_CONFIG.DATA_PATH "$OUT"
