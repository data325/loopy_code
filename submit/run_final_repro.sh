#!/usr/bin/env bash
# =============================================================================
# One-click reproduction of the final loopy submission
#   (quantile-axis per-axis calibration + VoxelRCNN, Combined 0.6512)
#   Data : quantile_axis -- per-axis 2001-quantile mapping
#   Detect: VoxelRCNN (GBlobs fork, 40 epochs, no test-time augmentation)
# All paths are relative to this directory. Derived/raw data are not shipped.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-urbantwin-v2xreal}"
DEVICE="${DEVICE:-0}"
PRED="${PREDICTIONS:-}"                 # optional: use provided predictions.json
SKIP_DETECT="${SKIP_DETECT:-0}"

TEST_FRAMES="$ROOT/competition/starting_kit/detection_test_frames"
FRAME_IDS="$ROOT/competition/starting_kit/detection_test_frame_ids.json"
FORBIDDEN="$ROOT/competition/starting_kit/forbidden_frames.txt"
BASE_SYN="${BASE_SYN:-$ROOT/data/base_synthetic}"
REAL_REF="${REAL_REF:-$ROOT/data/real_ref_frames}"
CKPT="$ROOT/third_party/GBlobs/checkpoints/voxelrcnn_final/checkpoint_epoch_40.pth"
CFG="$ROOT/configs/voxel_rcnn_gblobs_rangeldm2k_b_d50k_pedalign_v3.yaml"

OUT="$ROOT/outputs/final_repro"
SYN="$OUT/quantile_axis"
PY_RUN="conda run -n $ENV_NAME --no-capture-output python"
mkdir -p "$OUT"
cd "$ROOT"

echo "=============================================================="
echo " final submission reproduction | quantile_axis + VoxelRCNN | device=$DEVICE"
echo "=============================================================="

for p in "$TEST_FRAMES" "$FRAME_IDS" "$FORBIDDEN"; do
    [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done
if [ "$SKIP_DETECT" != "1" ]; then
    [ -f "$CKPT" ] || { echo "MISSING checkpoint: $CKPT"; exit 1; }
fi

if [ ! -f "$BASE_SYN/0001.bin" ] || [ ! -d "$REAL_REF" ]; then
    echo "MISSING prerequisite: $BASE_SYN/0001.bin or $REAL_REF"
    echo "  Derived/raw data are intentionally NOT shipped. See docs/FINAL_SUBMISSION_REPRODUCTION.md."
    exit 1
fi

echo "[1/3] generating quantile_axis frames (2001-quantile per-axis mapping)"
$PY_RUN src/generate_realism_baselines.py \
    --real-root "$REAL_REF" \
    --synthetic-root "$BASE_SYN" \
    --forbidden "$FORBIDDEN" \
    --output-root "$OUT" \
    --frames 50 --points-per-frame 2000 --seed 20260720
n_syn=$(ls "$SYN"/*.bin | wc -l)
echo "    quantile_axis frames: $n_syn x $(wc -c < "$SYN/0001.bin") bytes"

PRED_FILE="$OUT/predictions.json"
if [ "$SKIP_DETECT" = "1" ] || [ -n "$PRED" ]; then
    cp "$PRED" "$PRED_FILE"
    echo "[2/3] detection: using provided predictions $PRED"
else
    echo "[2/3] detection: VoxelRCNN (GBlobs, 40 epochs, no TTA)"
    GBLOBS=1 $PY_RUN src/final_submission_prepare.py infer-test \
        --config "$CFG" --checkpoint "$CKPT" \
        --frames "$TEST_FRAMES" --frame-ids "$FRAME_IDS" \
        --output "$PRED_FILE" --device "$DEVICE"
fi
n_box=$(python3 -c "import json;print(sum(len(v) for v in json.load(open('$PRED_FILE')).values()))")
echo "    prediction boxes: $n_box (submitted 10170)"

echo "[3/3] packaging"
cat > "$OUT/declaration.txt" <<EOF
I confirm that I did not train on any frame listed in forbidden_frames.txt,
and that all submitted detections were produced by a model trained only on
synthetic data I generated.

Team / individual name: loopy
Date (YYYY-MM-DD):       2026-08-15
Submission tag:          final
EOF
$PY_RUN src/build_submission.py \
    --synthetic-points "$SYN" --predictions "$PRED_FILE" \
    --frame-ids "$FRAME_IDS" --declaration "$OUT/declaration.txt" \
    --output "$OUT/submission_final_repro.zip"

echo "=============================================================="
echo " DONE: $OUT/submission_final_repro.zip"
echo " official reference: Combined 0.6512 | Detection 0.5583 | Realism 0.7905"
echo " prediction boxes: $n_box (submitted 10170: vehicle 4133 / pedestrian 3752 / truck 2285)"
[ "$n_box" = "10170" ] && echo " box-count check: PASS" || echo " box-count check: WARN (see docs/FINAL_SUBMISSION_REPRODUCTION.md)"
echo "=============================================================="
