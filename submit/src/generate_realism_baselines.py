#!/usr/bin/env python3
"""Generate dataset-level realism baselines from synthetic and allowed real data.

The challenge evaluates only the aggregate XYZ distribution.  Real frames whose
numeric suffix appears anywhere in forbidden_frames.txt are conservatively
excluded before fitting or validation, independent of scenario-index mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PC_RANGE = np.asarray([-40, -40, -8, 40, 40, 2], dtype=np.float32)


def stable_bucket(value: str, modulo: int = 5) -> int:
    return int.from_bytes(hashlib.sha1(value.encode()).digest()[:8], "big") % modulo


def load_xyz(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot infer width for {path}")
    points = points[:, :3]
    mask = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    mask &= (points >= PC_RANGE[:3]).all(axis=1) & (points <= PC_RANGE[3:]).all(axis=1)
    return np.asarray(points[mask], dtype=np.float32)


def collect_real_frames(root: Path, forbidden: Path) -> tuple[list[Path], list[Path]]:
    forbidden_suffixes = {
        line.strip().split("_")[-1]
        for line in forbidden.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    frames = []
    for scenario in sorted(path for path in root.iterdir() if path.is_dir()):
        infra = scenario / "-1"
        if not infra.is_dir():
            continue
        for path in sorted(infra.glob("*.bin")):
            if path.stem not in forbidden_suffixes:
                frames.append(path)
    fit = [path for path in frames if stable_bucket(str(path.relative_to(root))) != 0]
    validation = [path for path in frames if stable_bucket(str(path.relative_to(root))) == 0]
    return fit, validation


def pooled_points(paths: list[Path], max_frames: int, points_per_frame: int, seed: int) -> np.ndarray:
    if len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        paths = [paths[index] for index in indices]
    rng = np.random.default_rng(seed)
    parts = []
    for path in paths:
        points = load_xyz(path)
        if len(points) > points_per_frame:
            points = points[rng.choice(len(points), points_per_frame, replace=False)]
        parts.append(points)
    if not parts:
        raise ValueError("No usable point clouds")
    return np.concatenate(parts).astype(np.float32, copy=False)


def matrix_sqrt(covariance: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance.astype(np.float64))
    values = np.maximum(values, 1e-6)
    powers = 1.0 / np.sqrt(values) if inverse else np.sqrt(values)
    return (vectors * powers) @ vectors.T


def quantile_map(points: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    probabilities = np.linspace(0.0, 1.0, 2001)
    output = np.empty_like(points)
    for axis in range(3):
        source_knots = np.quantile(source[:, axis], probabilities)
        target_knots = np.quantile(target[:, axis], probabilities)
        output[:, axis] = np.interp(points[:, axis], source_knots, target_knots)
    return output


def write_candidate(root: Path, name: str, points: np.ndarray, frames: int, points_per_frame: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    required = frames * points_per_frame
    selected = points[rng.choice(len(points), required, replace=len(points) < required)].copy()
    selected = np.clip(selected, PC_RANGE[:3], PC_RANGE[3:]).astype(np.float32)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(selected.reshape(frames, points_per_frame, 3), 1):
        frame.tofile(directory / f"{index:04d}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--points-per-frame", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    real_fit_paths, real_validation_paths = collect_real_frames(args.real_root, args.forbidden)
    synthetic_paths = sorted(args.synthetic_root.glob("*.npy")) + sorted(args.synthetic_root.glob("*.bin"))
    real_fit = pooled_points(real_fit_paths, 320, 12000, args.seed)
    synthetic = pooled_points(synthetic_paths, 320, 12000, args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)
    base = synthetic[rng.choice(len(synthetic), args.frames * args.points_per_frame, replace=False)]

    candidates: dict[str, np.ndarray] = {"raw": base}
    syn_mean, real_mean = synthetic.mean(axis=0), real_fit.mean(axis=0)
    syn_std, real_std = synthetic.std(axis=0), real_fit.std(axis=0)
    candidates["affine_axis"] = (base - syn_mean) * (real_std / np.maximum(syn_std, 1e-6)) + real_mean

    syn_cov, real_cov = np.cov(synthetic, rowvar=False), np.cov(real_fit, rowvar=False)
    color = matrix_sqrt(syn_cov, inverse=True) @ matrix_sqrt(real_cov)
    candidates["affine_cov"] = (base - syn_mean) @ color + real_mean
    candidates["quantile_axis"] = quantile_map(base, synthetic, real_fit)

    gaussian = rng.multivariate_normal(real_mean, real_cov, size=len(base)).astype(np.float32)
    candidates["gaussian"] = gaussian

    jitter_scale = np.asarray([0.08, 0.08, 0.04], dtype=np.float32)
    empirical = real_fit[rng.choice(len(real_fit), len(base), replace=False)].copy()
    empirical += rng.normal(size=empirical.shape).astype(np.float32) * jitter_scale
    candidates["empirical_jitter"] = empirical

    args.output_root.mkdir(parents=True, exist_ok=True)
    for offset, (name, points) in enumerate(candidates.items()):
        write_candidate(args.output_root, name, points, args.frames, args.points_per_frame, args.seed + offset)

    manifest = {
        "real_fit_frames": len(real_fit_paths),
        "real_validation_frames": len(real_validation_paths),
        "synthetic_source_frames": len(synthetic_paths),
        "real_fit_points": len(real_fit),
        "synthetic_fit_points": len(synthetic),
        "forbidden_policy": "exclude every basename suffix occurring anywhere in forbidden_frames.txt",
        "validation_paths": [str(path) for path in real_validation_paths],
        "candidates": list(candidates),
        "frames": args.frames,
        "points_per_frame": args.points_per_frame,
        "seed": args.seed,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in manifest if key != "validation_paths"}, indent=2))


if __name__ == "__main__":
    main()
