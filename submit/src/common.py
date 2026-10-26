from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np


CHALLENGE_CLASSES = ("vehicle", "pedestrian", "truck")

# V2X-Real's three super-classes, extended with spelling variants found in
# synthetic sources. Bicycle/motorcycle-like actors follow the V2X-Real
# small-road-user grouping used for the pedestrian head.
SOURCE_CLASS_MAP = {
    "car": "vehicle",
    "policecar": "vehicle",
    "policevehicle": "vehicle",
    "longvehicle": "vehicle",
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "child": "pedestrian",
    "roadworker": "pedestrian",
    "scooter": "pedestrian",
    "scooterrider": "pedestrian",
    "motorcycle": "pedestrian",
    "motorcyclerider": "pedestrian",
    "motorcylerider": "pedestrian",
    "bicycle": "pedestrian",
    "bicyclerider": "pedestrian",
    "cyclist": "pedestrian",
    "truck": "truck",
    "boxtruck": "truck",
    "concretetruck": "truck",
    "van": "truck",
    "bus": "truck",
    "trashcan": "truck",
}


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def map_source_class(value: str) -> str | None:
    return SOURCE_CLASS_MAP.get(normalize_token(value))


def stable_fraction(value: str, seed: int = 1234) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def read_points(path: Path, binary_width: int | None = None) -> np.ndarray:
    """Read npy or raw float32 LiDAR and return finite Nx4 xyzi."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        points = np.asarray(np.load(path, mmap_mode="r"))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected NxC point array in {path}, got {points.shape}")
    elif path.suffix.lower() == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        width = binary_width or infer_binary_width(raw)
        points = raw.reshape(-1, width)
    else:
        raise ValueError(f"Unsupported point-cloud extension: {path}")

    points = np.asarray(points, dtype=np.float32)
    if points.shape[1] == 3:
        points = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
    else:
        points = points[:, :4].copy()
    return points[np.isfinite(points).all(axis=1)]


def infer_binary_width(raw: np.ndarray) -> int:
    candidates = [w for w in (3, 4) if raw.size and raw.size % w == 0]
    if not candidates:
        raise ValueError(f"Raw float count {raw.size} is not divisible by 3 or 4")
    if len(candidates) == 1:
        return candidates[0]

    def score(width: int) -> float:
        pts = raw.reshape(-1, width)
        xyz = pts[:, :3]
        finite = np.isfinite(xyz).all(axis=1).mean()
        plausible = (np.abs(xyz) < 500).all(axis=1).mean()
        spread = np.percentile(xyz, 95, axis=0) - np.percentile(xyz, 5, axis=0)
        vertical_ratio = spread[2] / max(spread[0], spread[1], 1e-6)
        intensity_bonus = 0.0
        if width == 4:
            intensity = pts[:, 3]
            intensity_bonus = 0.2 * np.mean((intensity >= 0) & (intensity <= 255))
        return 3 * finite + plausible - min(vertical_ratio, 2.0) + intensity_bonus

    return max(candidates, key=score)


def parse_detection_label_line(line: str) -> tuple[np.ndarray, str] | None:
    """Parse common UrbanTwin/OpenPCDet label layouts into box + class."""
    tokens = [x for x in re.split(r"[\s,;]+", line.strip()) if x]
    if not tokens or tokens[0].startswith("#"):
        return None

    mapped = None
    for token in tokens:
        mapped = map_source_class(token)
        if mapped:
            break
    if mapped is None:
        return None

    numbers: list[float] = []
    for token in tokens:
        try:
            numbers.append(float(token))
        except ValueError:
            pass
    if len(numbers) < 7:
        raise ValueError(f"Detection label has fewer than 7 numeric values: {line!r}")

    box = np.asarray(numbers[:7], dtype=np.float32)
    if abs(float(box[6])) > 2 * np.pi + 0.1:
        box[6] = np.deg2rad(box[6])
    if np.any(box[3:6] <= 0):
        raise ValueError(f"Non-positive box extent: {line!r}")
    return box, mapped


def read_detection_labels(path: Path) -> list[tuple[np.ndarray, str]]:
    labels = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                parsed = parse_detection_label_line(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if parsed is not None:
                labels.append(parsed)
    return labels


def write_detection_labels(path: Path, labels: list[tuple[np.ndarray, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for box, name in labels:
            values = " ".join(f"{float(v):.7g}" for v in box[:7])
            handle.write(f"{values} {name}\n")

