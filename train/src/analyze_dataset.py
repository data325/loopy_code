#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import CHALLENGE_CLASSES, read_detection_labels, read_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OpenPCDet custom UrbanTwin data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ids = (args.data / "ImageSets" / "train.txt").read_text().split()
    counts: Counter[str] = Counter()
    sizes: dict[str, list[np.ndarray]] = defaultdict(list)
    bottoms: dict[str, list[float]] = defaultdict(list)
    point_counts = []
    for sample_id in ids:
        points = read_points(args.data / "points" / f"{sample_id}.npy")
        point_counts.append(len(points))
        for box, name in read_detection_labels(args.data / "labels" / f"{sample_id}.txt"):
            counts[name] += 1
            sizes[name].append(box[3:6])
            bottoms[name].append(float(box[2] - box[5] / 2))

    summary = {
        "frames": len(ids),
        "points_per_frame": {
            "mean": float(np.mean(point_counts)),
            "median": float(np.median(point_counts)),
            "p05": float(np.percentile(point_counts, 5)),
            "p95": float(np.percentile(point_counts, 95)),
        },
        "classes": {},
    }
    for name in CHALLENGE_CLASSES:
        arr = np.asarray(sizes[name])
        summary["classes"][name] = {
            "count": counts[name],
            "median_size_dx_dy_dz": np.median(arr, axis=0).tolist() if len(arr) else None,
            "median_bottom_z": float(np.median(bottoms[name])) if bottoms[name] else None,
        }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

