#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from common import CHALLENGE_CLASSES, read_points


RANGE_MIN = np.array([-40, -40, -8], dtype=np.float32)
RANGE_MAX = np.array([40, 40, 2], dtype=np.float32)
HONOR_STATEMENT = (
    "I confirm that I did not train on any frame listed in forbidden_frames.txt, "
    "and that all submitted detections were produced by a model trained only on synthetic data I generated."
)


def validate_predictions(predictions: dict, frame_ids: list[str]) -> None:
    if set(predictions) != set(frame_ids):
        missing = sorted(set(frame_ids) - set(predictions))
        extra = sorted(set(predictions) - set(frame_ids))
        raise ValueError(f"Prediction keys mismatch. missing={missing[:5]}, extra={extra[:5]}")
    for frame_id, rows in predictions.items():
        if not isinstance(rows, list):
            raise ValueError(f"{frame_id}: predictions must be a list")
        for row in rows:
            if row.get("class", "").lower() not in CHALLENGE_CLASSES:
                raise ValueError(f"{frame_id}: invalid class {row.get('class')!r}")
            box = np.asarray(row.get("box", []), dtype=float)
            if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                raise ValueError(f"{frame_id}: invalid box {row.get('box')!r}")
            score = float(row.get("score", -1))
            if not 0 <= score <= 1:
                raise ValueError(f"{frame_id}: score outside [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and build Codabench submission.zip")
    parser.add_argument("--synthetic-points", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frame-ids", type=Path, required=True)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-unsigned-draft", action="store_true")
    args = parser.parse_args()

    ids_obj = json.loads(args.frame_ids.read_text(encoding="utf-8"))
    frame_ids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    frame_ids = [str(x) for x in frame_ids]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    validate_predictions(predictions, frame_ids)

    candidates = sorted([p for p in args.synthetic_points.iterdir() if p.suffix.lower() in {".npy", ".bin"}])
    if len(candidates) < 50:
        raise ValueError(f"Need at least 50 synthetic frames, found {len(candidates)}")
    # Deterministic spread over the source sequence avoids selecting a short,
    # highly correlated temporal window.
    indexes = np.linspace(0, len(candidates) - 1, 50, dtype=int)
    selected = [candidates[i] for i in indexes]
    rng = np.random.default_rng(args.seed)

    if args.declaration is None and not args.allow_unsigned_draft:
        raise ValueError("A signed declaration is required (or pass --allow-unsigned-draft for format testing only)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="urbantwin_submission_") as temp_name:
        temp = Path(temp_name)
        synthetic_dir = temp / "synthetic"
        synthetic_dir.mkdir()
        for index, source in enumerate(selected, 1):
            points = read_points(source)[:, :3]
            keep = ((points >= RANGE_MIN) & (points <= RANGE_MAX)).all(axis=1)
            points = points[keep]
            if len(points) > 115_000:
                points = points[rng.choice(len(points), 115_000, replace=False)]
            points.astype(np.float32).tofile(synthetic_dir / f"{index:04d}.bin")
        shutil.copy2(args.predictions, temp / "predictions.json")
        if args.declaration:
            suffix = args.declaration.suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                raise ValueError("Declaration must be PDF or TXT")
            shutil.copy2(args.declaration, temp / f"declaration{suffix}")
        else:
            (temp / "declaration.txt").write_text(
                "UNSIGNED DRAFT - DO NOT SUBMIT\n\n" + HONOR_STATEMENT + "\n",
                encoding="utf-8",
            )
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(temp.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(temp).as_posix())
    print(f"Built {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()

