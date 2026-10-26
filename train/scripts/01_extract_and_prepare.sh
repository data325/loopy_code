#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PARTS="$PROJECT/data/urbantwin_parts"
RAW="$PROJECT/data/urbantwin_raw"
CUSTOM="$PROJECT/data/openpcdet_custom"

mkdir -p "$RAW" "$CUSTOM"
if [[ ! -f "$PARTS/11535585" ]]; then
  echo "UrbanTwin split archives are missing under $PARTS" >&2
  exit 1
fi

7z x "$PARTS/11535585" -o"$RAW" -y
conda run -n urbantwin-v2xreal python "$PROJECT/src/prepare_dataset.py" \
  --raw-root "$RAW" --output "$CUSTOM" --seed 1234 --val-fraction 0.2
conda run -n urbantwin-v2xreal python "$PROJECT/src/analyze_dataset.py" \
  --data "$CUSTOM" --output "$PROJECT/outputs/dataset_stats.json"

cd "$PROJECT/third_party/OpenPCDet"
conda run -n urbantwin-v2xreal python -m pcdet.datasets.custom.custom_dataset \
  create_custom_infos "$PROJECT/configs/second_urbantwin.yaml"

