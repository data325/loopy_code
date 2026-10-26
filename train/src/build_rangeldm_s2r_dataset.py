#!/usr/bin/env python3
"""Build paired-format, unpaired-domain range images for RangeLDM-S2R.

Real V2X-Real scans are stored as native organized 128x1024 returns. UrbanTwin
points are projected onto the same beam table. The output keeps source geometry,
return masks, geometric cues, and box-derived class channels.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml


H, W = 128, 1024
CLASSES = {"vehicle": 0, "pedestrian": 1, "truck": 2}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "V2X-Real"))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils import box_utils  # noqa: E402

INVERSE_CLASS_MAP = {fine: coarse for coarse, names in SUPER_CLASS_MAP.items() for fine in names}
PC_RANGE = [-40, -40, -8, 40, 40, 2]


def valid_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(1) & np.any(xyz != 0, axis=1)


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    boxes, names = [], []
    for line in path.read_text().splitlines():
        row = line.split()
        if len(row) != 8 or row[-1] not in CLASSES:
            continue
        boxes.append([float(v) for v in row[:7]])
        names.append(CLASSES[row[-1]])
    return np.asarray(boxes, np.float32).reshape(-1, 7), np.asarray(names, np.int64)


def points_in_boxes(xyz: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return N x B box-membership without CUDA dependencies."""
    if not len(boxes):
        return np.zeros((len(xyz), 0), dtype=bool)
    out = np.zeros((len(xyz), len(boxes)), dtype=bool)
    for index, box in enumerate(boxes):
        rel = xyz - box[None, :3]
        c, s = np.cos(box[6]), np.sin(box[6])
        local_x = c * rel[:, 0] + s * rel[:, 1]
        local_y = -s * rel[:, 0] + c * rel[:, 1]
        out[:, index] = (
            (np.abs(local_x) <= box[3] / 2)
            & (np.abs(local_y) <= box[4] / 2)
            & (np.abs(rel[:, 2]) <= box[5] / 2)
        )
    return out


def infer_native_layout(raw: np.ndarray) -> np.ndarray:
    """Choose HxW orientation whose rows best match physical laser rings."""
    candidates = [raw.reshape(H, W, 4), raw.reshape(W, H, 4).transpose(1, 0, 2)]
    scores = []
    for grid in candidates:
        xyz = grid[..., :3]
        rho = np.linalg.norm(xyz[..., :2], axis=-1)
        elev = np.arctan2(xyz[..., 2], rho)
        mask = np.any(xyz != 0, axis=-1)
        row_var = []
        for row in range(H):
            values = elev[row][mask[row]]
            row_var.append(np.var(values) if len(values) else 1.0)
        scores.append(float(np.median(row_var)))
    return candidates[int(np.argmin(scores))]


def real_boxes_from_yaml(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    grouped = {}
    for object_id, obj in metadata.get("vehicles", {}).items():
        super_name = INVERSE_CLASS_MAP.get(obj.get("obj_type"))
        if super_name not in CLASSES:
            continue
        item = copy.deepcopy(obj)
        item["obj_type"] = np.asarray([super_name])
        grouped[object_id] = item
    projected = {}
    box_utils.project_world_objects(grouped, projected, metadata["lidar_pose"], PC_RANGE, "lwh")
    boxes, classes = [], []
    for box_and_class in projected.values():
        row = box_and_class[0]
        boxes.append(np.asarray(row[:7], np.float32))
        classes.append(CLASSES[str(row[7]).lower()])
    return np.asarray(boxes, np.float32).reshape(-1, 7), np.asarray(classes, np.int64)


def metadata_for_real(path: Path, metadata_root: Path) -> Path:
    resolved = path.resolve()
    train_part = next(x for x in resolved.parts if x.startswith("train") and x[5:x.index("_agent_")].isdigit())
    train_name, agent = train_part.split("_agent_", 1)
    return metadata_root / f"{train_name}_agent_{agent}_yaml" / resolved.parent.parent.name / resolved.parent.name / f"{resolved.stem}.yaml"


def native_real(path: Path, boxes=None, box_classes=None) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, np.float32)
    if raw.size != H * W * 4:
        raise ValueError(f"{path}: expected {H*W} four-column rays, got {raw.size//4}")
    grid = infer_native_layout(raw)
    xyz = grid[..., :3].astype(np.float32)
    mask = np.any(xyz != 0, axis=-1) & np.isfinite(xyz).all(axis=-1)
    ranges = np.linalg.norm(xyz, axis=-1).astype(np.float32)
    intensity = grid[..., 3].astype(np.float32)
    semantic = None
    if boxes is not None and box_classes is not None:
        flat = xyz.reshape(-1, 3)
        membership = points_in_boxes(flat, boxes)
        semantic = np.zeros((len(flat), 3), np.float32)
        for box_index, class_index in enumerate(box_classes):
            semantic[membership[:, box_index], class_index] = 1
        semantic = semantic.reshape(H, W, 3)
        semantic[~mask] = 0
    return make_features(xyz, ranges, mask, intensity=intensity, semantic=semantic)


def synthetic_project(
    points: np.ndarray,
    ring_angles: np.ndarray,
    boxes: np.ndarray,
    box_classes: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    xyz = points[:, :3].astype(np.float32)
    keep = valid_xyz(xyz)
    xyz, intensity = xyz[keep], points[keep, 3].astype(np.float32)
    ranges = np.linalg.norm(xyz, axis=1)
    rho = np.linalg.norm(xyz[:, :2], axis=1)
    elev = np.arctan2(xyz[:, 2], rho)
    yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
    rows = np.abs(elev[:, None] - ring_angles[None]).argmin(1)
    cols = np.floor((yaw + np.pi) / (2 * np.pi) * W).astype(np.int64) % W

    # Sort far-to-near so the nearest return wins on repeated ray indices.
    order = np.argsort(ranges)[::-1]
    rows, cols = rows[order], cols[order]
    xyz, ranges, intensity = xyz[order], ranges[order], intensity[order]
    membership = points_in_boxes(xyz, boxes)
    semantic = np.zeros((len(xyz), 3), np.float32)
    for box_index, class_index in enumerate(box_classes):
        semantic[membership[:, box_index], class_index] = 1

    xyz_grid = np.zeros((H, W, 3), np.float32)
    range_grid = np.zeros((H, W), np.float32)
    intensity_grid = np.zeros((H, W), np.float32)
    semantic_grid = np.zeros((H, W, 3), np.float32)
    source_index = np.full((H, W), -1, np.int64)
    xyz_grid[rows, cols] = xyz
    range_grid[rows, cols] = ranges
    intensity_grid[rows, cols] = intensity
    semantic_grid[rows, cols] = semantic
    source_index[rows, cols] = order
    mask = range_grid > 0

    # Round-trip through the target beam table.
    az = ((np.arange(W) + 0.5) / W * 2 * np.pi - np.pi)[None, :]
    el = ring_angles[:, None]
    reconstructed = np.stack(
        [
            range_grid * np.cos(el) * np.cos(az),
            range_grid * np.cos(el) * np.sin(az),
            np.broadcast_to(range_grid * np.sin(el), (H, W)),
        ],
        axis=-1,
    )
    error = np.linalg.norm(reconstructed[mask] - xyz_grid[mask], axis=1)
    object_input = int(membership.any(1).sum())
    object_retained = int((semantic_grid.sum(-1) > 0).sum())
    audit = {
        "input_points": int(len(xyz)),
        "retained_rays": int(mask.sum()),
        "retention_rate": float(mask.sum() / max(len(xyz), 1)),
        "roundtrip_median_m": float(np.median(error)) if len(error) else float("inf"),
        "roundtrip_p95_m": float(np.percentile(error, 95)) if len(error) else float("inf"),
        "object_points_input": object_input,
        "object_rays_retained": object_retained,
        "object_retention_rate": float(object_retained / max(object_input, 1)),
    }
    features = make_features(
        reconstructed.astype(np.float32),
        range_grid,
        mask,
        intensity=intensity_grid,
        semantic=semantic_grid,
    )
    return features, audit


def make_features(
    xyz: np.ndarray,
    ranges: np.ndarray,
    mask: np.ndarray,
    intensity: np.ndarray | None = None,
    semantic: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    log_range = np.zeros_like(ranges, np.float32)
    log_range[mask] = np.log2(ranges[mask] + 1) / 8.0
    left = np.roll(xyz, 1, axis=1)
    right = np.roll(xyz, -1, axis=1)
    up = np.roll(xyz, 1, axis=0)
    down = np.roll(xyz, -1, axis=0)
    normals = np.cross(right - left, down - up)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-6)
    normals[~mask] = 0
    range_edge = np.maximum(
        np.abs(ranges - np.roll(ranges, 1, 1)),
        np.abs(ranges - np.roll(ranges, -1, 1)),
    )
    range_edge = np.clip(range_edge / 10.0, 0, 1).astype(np.float32)
    range_edge[~mask] = 0
    payload = {
        "range": ranges.astype(np.float32),
        "log_range": log_range,
        "mask": mask.astype(np.float32),
        "xyz": xyz.astype(np.float32),
        "normal": normals.astype(np.float32),
        "edge": range_edge,
    }
    if intensity is not None:
        payload["intensity"] = intensity.astype(np.float32)
    if semantic is not None:
        payload["semantic"] = semantic.astype(np.float32)
        boundary = np.zeros(mask.shape, dtype=bool)
        occupied = semantic.sum(-1) > 0
        boundary |= occupied != np.roll(occupied, 1, 1)
        boundary |= occupied != np.roll(occupied, -1, 1)
        boundary |= occupied != np.roll(occupied, 1, 0)
        boundary |= occupied != np.roll(occupied, -1, 0)
        payload["object_boundary"] = boundary.astype(np.float32)
    return payload


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-metadata-root", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-frames", type=int, default=64)
    parser.add_argument("--real-frames", type=int, default=64)
    args = parser.parse_args()

    ring_angles = np.load(args.profile / "ring_angles.npy").astype(np.float32)
    if len(ring_angles) != H:
        raise ValueError(f"expected {H} ring angles, got {len(ring_angles)}")
    forbidden = {
        line.strip().split("_")[-1]
        for line in args.forbidden.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    syn_ids = (args.synthetic_root / "ImageSets/train.txt").read_text().split()
    syn_ids = syn_ids[: args.synthetic_frames]
    audits = []
    for frame_id in syn_ids:
        points = np.load(args.synthetic_root / "points" / f"{frame_id}.npy")
        boxes, classes = load_labels(args.synthetic_root / "labels" / f"{frame_id}.txt")
        payload, audit = synthetic_project(points, ring_angles, boxes, classes)
        save_npz(args.output / "synthetic" / f"{frame_id}.npz", payload)
        audits.append({"frame_id": frame_id, **audit})

    real_paths = [
        p for p in sorted(args.real_root.glob("**/-1/*.bin")) if p.stem not in forbidden
    ]
    if len(real_paths) > args.real_frames:
        indices = np.linspace(0, len(real_paths) - 1, args.real_frames, dtype=int)
        real_paths = [real_paths[i] for i in indices]
    real_records = []
    for index, path in enumerate(real_paths):
        yaml_path = metadata_for_real(path, args.real_metadata_root) if args.real_metadata_root else None
        boxes = classes = None
        if yaml_path is not None:
            if not yaml_path.exists():
                raise FileNotFoundError(f"missing metadata for {path}: {yaml_path}")
            boxes, classes = real_boxes_from_yaml(yaml_path)
        payload = native_real(path, boxes, classes)
        name = f"{index:06d}_{path.parent.parent.name}_{path.stem}.npz"
        save_npz(args.output / "real" / name, payload)
        real_records.append({
            "output": name,
            "source": str(path),
            "valid_returns": int(payload["mask"].sum()),
            "semantic_rays": int(payload.get("semantic", np.zeros(1)).sum()),
            "metadata": str(yaml_path) if yaml_path else None,
        })

    summary = {
        "format": "RangeLDM-S2R M0 128x1024",
        "synthetic_frames": len(audits),
        "real_frames": len(real_records),
        "forbidden_suffixes_excluded": len(forbidden),
        "synthetic_summary": {
            key: float(np.median([row[key] for row in audits]))
            for key in (
                "retention_rate",
                "roundtrip_median_m",
                "roundtrip_p95_m",
                "object_retention_rate",
            )
        },
        "synthetic_audit": audits,
        "real_records": real_records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"synthetic_audit", "real_records"}}, indent=2))


if __name__ == "__main__":
    main()
