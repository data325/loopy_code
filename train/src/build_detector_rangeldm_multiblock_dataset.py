#!/usr/bin/env python3
"""Build an OpenPCDet dataset from multiple non-overlapping RangeLDM blocks."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def symlink(source: Path, target: Path) -> None:
    target.symlink_to(source.resolve())


def parse_block(value: str) -> tuple[str, Path, Path]:
    name, source_list, generated = value.split(":", 2)
    if not name.isidentifier():
        raise ValueError(f"invalid block name: {name}")
    return name, Path(source_list), Path(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--info-root", type=Path, required=True)
    parser.add_argument("--block", action="append", required=True,
                        help="NAME:SOURCE_IDS_TXT:GENERATED_ROOT; may be repeated")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    blocks = [parse_block(value) for value in args.block]
    base_train_ids = (args.base_root / "ImageSets" / "train.txt").read_text().split()
    base_val_ids = (args.base_root / "ImageSets" / "val.txt").read_text().split()
    train_infos = load_pickle(args.info_root / "custom_infos_train_compat.pkl")[:len(base_train_ids)]
    val_infos = load_pickle(args.info_root / "custom_infos_val_compat.pkl")
    if len(train_infos) != len(base_train_ids) or len(val_infos) != len(base_val_ids):
        raise ValueError("base info/image-set count mismatch")
    info_by_id = {item["point_cloud"]["lidar_idx"]: item for item in train_infos}

    resolved = []
    seen_sources: set[str] = set()
    for name, source_list, generated_root in blocks:
        source_ids = source_list.read_text().split()
        generated = {path.stem: path for path in generated_root.rglob("*.npy")}
        if len(source_ids) != len(set(source_ids)) or seen_sources.intersection(source_ids):
            raise ValueError(f"source IDs are duplicated in block {name}")
        seen_sources.update(source_ids)
        missing = [frame_id for frame_id in source_ids if frame_id not in generated or frame_id not in info_by_id]
        if missing or len(generated) != len(source_ids):
            raise ValueError(f"block {name} mismatch: generated={len(generated)}, ids={len(source_ids)}, missing={missing[:3]}")
        resolved.append((name, source_ids, generated))

    points_out = args.output_root / "points"
    labels_out = args.output_root / "labels"
    imagesets_out = args.output_root / "ImageSets"
    points_out.mkdir(parents=True)
    labels_out.mkdir()
    imagesets_out.mkdir()
    for frame_id in sorted(set(base_train_ids) | set(base_val_ids)):
        symlink(args.base_root / "points" / f"{frame_id}.npy", points_out / f"{frame_id}.npy")
        symlink(args.base_root / "labels" / f"{frame_id}.txt", labels_out / f"{frame_id}.txt")

    augmented_infos, augmented_ids, mapping = [], [], []
    for name, source_ids, generated in resolved:
        for index, source_id in enumerate(source_ids):
            augmented_id = f"{name}{index:06d}"
            xyz = np.load(generated[source_id]).astype(np.float32)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                raise ValueError(f"invalid points for {name}:{source_id}")
            np.save(points_out / f"{augmented_id}.npy", xyz)
            symlink(args.base_root / "labels" / f"{source_id}.txt", labels_out / f"{augmented_id}.txt")
            item = copy.deepcopy(info_by_id[source_id])
            item["point_cloud"]["lidar_idx"] = augmented_id
            augmented_infos.append(item)
            augmented_ids.append(augmented_id)
            mapping.append({"block": name, "augmented_id": augmented_id, "source_id": source_id, "points": int(len(xyz))})

    (imagesets_out / "train.txt").write_text("\n".join(base_train_ids + augmented_ids) + "\n")
    (imagesets_out / "val.txt").write_text("\n".join(base_val_ids) + "\n")
    dump_pickle(train_infos + augmented_infos, args.output_root / "custom_infos_train_compat.pkl")
    dump_pickle(val_infos, args.output_root / "custom_infos_val_compat.pkl")
    report = {
        "base_train_frames": len(base_train_ids),
        "base_val_frames": len(base_val_ids),
        "augmented_frames": len(augmented_ids),
        "blocks": {name: len(ids) for name, ids, _ in resolved},
        "point_count_median": float(np.median([item["points"] for item in mapping])),
        "mapping": mapping,
    }
    (args.output_root / "rangeldm_multiblock_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "mapping"}, indent=2))


if __name__ == "__main__":
    main()
