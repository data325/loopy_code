#!/usr/bin/env python3
"""Build leakage-resistant temporal group splits for the 10K UrbanTwin sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def bucket(group_id: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", default="UCF-DT-V2X-Real-Seq-1")
    parser.add_argument("--group-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    ids = sorted(p.stem for p in (args.source / "points").glob("*.npy") if int(p.stem) < 10000)
    if len(ids) != 10000:
        raise ValueError(f"Expected 10,000 original frames, found {len(ids)}")

    splits = {"train": [], "scenario_val": [], "sensor_stress_val": []}
    groups = {}
    for sample_id in ids:
        group_id = f"{args.sequence}:block-{int(sample_id) // args.group_size:03d}"
        value = bucket(group_id, args.seed)
        split = "train" if value < 8 else ("scenario_val" if value == 8 else "sensor_stress_val")
        splits[split].append(sample_id)
        groups.setdefault(group_id, {"split": split, "ids": []})["ids"].append(sample_id)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (args.output / f"{name}.txt").write_text("\n".join(values) + "\n")
    # OpenPCDet uses train/val names. Scenario-val is the primary validation.
    (args.output / "val.txt").write_text("\n".join(splits["scenario_val"]) + "\n")
    manifest = {
        "method": "contiguous temporal blocks assigned by stable group hash",
        "sequence": args.sequence,
        "group_size": args.group_size,
        "seed": args.seed,
        "ellipsoid_augmented_ids_excluded": "010000-011999",
        "counts": {key: len(value) for key, value in splits.items()},
        "groups": groups,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**manifest, "groups": len(groups)}, indent=2))


if __name__ == "__main__":
    main()
