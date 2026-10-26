#!/usr/bin/env python3
"""Build an OpenPCDet dataset from a non-overlapping anchored RangeLDM block."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(target)
        return
    target.symlink_to(source.resolve())


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--info-root", type=Path, required=True,
                        help="Compatible base info files; first 8K entries are retained.")
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_ids = args.source_list.read_text().split()
    generated = sorted(args.generated.rglob("*.npy"))
    generated_by_id = {p.stem: p for p in generated}
    missing = [frame_id for frame_id in source_ids if frame_id not in generated_by_id]
    if missing or len(generated_by_id) != len(source_ids):
        raise ValueError(
            f"generated/source mismatch: generated={len(generated_by_id)}, "
            f"source={len(source_ids)}, missing={missing[:3]}"
        )
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    base_train_ids = (args.base_root / "ImageSets" / "train.txt").read_text().split()
    base_val_ids = (args.base_root / "ImageSets" / "val.txt").read_text().split()
    base_all_ids = sorted(set(base_train_ids) | set(base_val_ids))
    base_train_infos = load_pickle(args.info_root / "custom_infos_train_compat.pkl")
    base_val_infos = load_pickle(args.info_root / "custom_infos_val_compat.pkl")
    if len(base_train_infos) < len(base_train_ids) or len(base_val_infos) != len(base_val_ids):
        raise ValueError("base info/image-set count mismatch")
    # info_root may be the prior 8K+2K experiment; retain only its original 8K prefix.
    base_train_infos = base_train_infos[:len(base_train_ids)]
    info_by_id = {item["point_cloud"]["lidar_idx"]: item for item in base_train_infos}
    if any(frame_id not in info_by_id for frame_id in source_ids):
        raise ValueError("a selected source frame is absent from base training infos")

    points_out = args.output_root / "points"
    labels_out = args.output_root / "labels"
    imagesets_out = args.output_root / "ImageSets"
    points_out.mkdir(parents=True)
    labels_out.mkdir()
    imagesets_out.mkdir()
    for frame_id in base_all_ids:
        symlink(args.base_root / "points" / f"{frame_id}.npy", points_out / f"{frame_id}.npy")
        symlink(args.base_root / "labels" / f"{frame_id}.txt", labels_out / f"{frame_id}.txt")

    augmented_infos = []
    mapping = []
    for index, source_id in enumerate(source_ids):
        augmented_id = f"r{index:06d}"
        xyz = np.load(generated_by_id[source_id]).astype(np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError(f"invalid generated point cloud: {generated_by_id[source_id]}")
        # The active DSVT uses xyz only; keep four columns to preserve the dataset contract.
        points = np.concatenate([xyz, np.zeros((len(xyz), 1), np.float32)], axis=1)
        np.save(points_out / f"{augmented_id}.npy", points)
        symlink(args.base_root / "labels" / f"{source_id}.txt", labels_out / f"{augmented_id}.txt")
        item = copy.deepcopy(info_by_id[source_id])
        item["point_cloud"]["lidar_idx"] = augmented_id
        augmented_infos.append(item)
        mapping.append({"augmented_id": augmented_id, "source_id": source_id, "points": int(len(points))})

    train_ids = base_train_ids + [item["augmented_id"] for item in mapping]
    (imagesets_out / "train.txt").write_text("\n".join(train_ids) + "\n")
    (imagesets_out / "val.txt").write_text("\n".join(base_val_ids) + "\n")
    dump_pickle(base_train_infos + augmented_infos, args.output_root / "custom_infos_train_compat.pkl")
    dump_pickle(base_val_infos, args.output_root / "custom_infos_val_compat.pkl")
    report = {
        "base_train_frames": len(base_train_ids),
        "base_val_frames": len(base_val_ids),
        "augmented_frames": len(mapping),
        "point_count_median": float(np.median([item["points"] for item in mapping])),
        "source_id_first": source_ids[0],
        "source_id_last": source_ids[-1],
        "mapping": mapping,
    }
    (args.output_root / "rangeldm_block_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "mapping"}, indent=2))


if __name__ == "__main__":
    main()
