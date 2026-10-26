#!/usr/bin/env python3
"""Prepare a non-overlapping slice of synthetic frames for RangeLDM."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    ids = (args.source_root / "ImageSets" / "train.txt").read_text().split()
    selected = ids[args.start : args.start + args.count]
    if len(selected) != args.count:
        raise ValueError(f"requested {args.count} frames at {args.start}, found {len(selected)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    image_sets = args.output_root / "ImageSets"
    image_sets.mkdir(exist_ok=True)
    (image_sets / "train.txt").write_text("\n".join(selected) + "\n")
    for name in ("points", "labels"):
        target = args.output_root / name
        if target.exists() or target.is_symlink():
            if target.resolve() != (args.source_root / name).resolve():
                raise FileExistsError(target)
        else:
            target.symlink_to((args.source_root / name).resolve(), target_is_directory=True)
    print(f"prepared {len(selected)} frames: {selected[0]}..{selected[-1]}")


if __name__ == "__main__":
    main()
