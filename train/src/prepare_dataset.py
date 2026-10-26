#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import read_detection_labels, read_points, stable_fraction, write_detection_labels


POINT_DIR_NAMES = {"lidar", "lidars", "point", "points", "pointcloud", "pointclouds", "pc", "pcd", "velodyne"}
LABEL_DIR_NAMES = {"lbl", "label", "labels", "annotation", "annotations", "bbox", "boxes"}
EXCLUDED_POINT_PARTS = {"ins_seg", "sem_seg", "instances", "semantic", "segmentation"}


def is_under_named_dir(path: Path, names: set[str]) -> bool:
    return any(part.lower().replace("_", "") in {n.replace("_", "") for n in names} for part in path.parts)


def discover_pairs(raw_root: Path) -> list[tuple[Path, Path]]:
    point_files = [
        p for p in raw_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".npy", ".bin"}
        and is_under_named_dir(p.parent, POINT_DIR_NAMES)
        and not any(part.lower() in EXCLUDED_POINT_PARTS for part in p.parts)
    ]
    label_files = [
        p for p in raw_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".lbl"}
        and is_under_named_dir(p.parent, LABEL_DIR_NAMES)
    ]
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in label_files:
        labels_by_stem[path.stem].append(path)

    pairs = []
    for point_path in sorted(point_files):
        candidates = labels_by_stem.get(point_path.stem, [])
        if not candidates:
            continue
        label_path = min(candidates, key=lambda x: len(set(x.parts) ^ set(point_path.parts)))
        pairs.append((point_path, label_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UrbanTwin release to OpenPCDet CustomDataset format")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--copy", action="store_true", help="Copy points instead of writing normalized npy files")
    args = parser.parse_args()

    pairs = discover_pairs(args.raw_root)
    if not pairs:
        raise SystemExit(f"No point/label pairs discovered below {args.raw_root}")

    points_dir = args.output / "points"
    labels_dir = args.output / "labels"
    imagesets_dir = args.output / "ImageSets"
    for directory in (points_dir, labels_dir, imagesets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": []}
    class_counts: Counter[str] = Counter()
    skipped_empty = 0
    source_manifest = []
    for index, (point_path, label_path) in enumerate(pairs):
        sample_id = f"{index:06d}"
        labels = read_detection_labels(label_path)
        if not labels:
            skipped_empty += 1
        class_counts.update(name for _, name in labels)

        output_points = points_dir / f"{sample_id}.npy"
        if args.copy and point_path.suffix.lower() == ".npy":
            shutil.copy2(point_path, output_points)
        else:
            np.save(output_points, read_points(point_path))
        write_detection_labels(labels_dir / f"{sample_id}.txt", labels)

        split = "val" if stable_fraction(str(point_path.relative_to(args.raw_root)), args.seed) < args.val_fraction else "train"
        splits[split].append(sample_id)
        source_manifest.append({
            "sample_id": sample_id,
            "point_path": str(point_path.relative_to(args.raw_root)),
            "label_path": str(label_path.relative_to(args.raw_root)),
            "split": split,
        })

    for split, ids in splits.items():
        (imagesets_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (args.output / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    summary = {
        "samples": len(pairs),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "empty_label_frames": skipped_empty,
        "class_counts": dict(class_counts),
        "seed": args.seed,
    }
    (args.output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

