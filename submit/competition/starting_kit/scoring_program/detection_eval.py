"""
Multi-class 3D object detection evaluation for the UrbanTwin sim2real
challenge.

Computes per-class average precision over 3D IoU (or BEV IoU) at one or
more thresholds, then mean AP across classes. Single-class evaluation is
the multi-class evaluation with one entry in `class_names`.

Box convention: 7-DoF axis-aligned-in-Z bounding box
    [x, y, z, dx, dy, dz, heading]
where (x, y, z) is the box center, (dx, dy, dz) are extents along the
box-local x/y/z axes, and heading is rotation around +Z in radians.
This matches OpenPCDet's `gt_boxes` format.

The KITTI Easy/Mod/Hard difficulty filter is intentionally not applied:
those depend on 2D bounding-box truncation/occlusion/min-pixel-height,
which are not natural for roadside-LiDAR-only annotations.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def _box_corners_2d(box: np.ndarray) -> np.ndarray:
    """Return the 4 BEV corners of a 7-DoF box, shape (4, 2)."""
    x, y, _, dx, dy, _, heading = box[:7]
    cos, sin = np.cos(heading), np.sin(heading)
    half_dx, half_dy = dx / 2.0, dy / 2.0
    corners = np.array([
        [ half_dx,  half_dy],
        [ half_dx, -half_dy],
        [-half_dx, -half_dy],
        [-half_dx,  half_dy],
    ])
    rot = np.array([[cos, -sin], [sin, cos]])
    return corners @ rot.T + np.array([x, y])


def _bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = Polygon(_box_corners_2d(a))
    pb = Polygon(_box_corners_2d(b))
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 0 else 0.0


def _z_overlap(a: np.ndarray, b: np.ndarray) -> float:
    az_lo, az_hi = a[2] - a[5] / 2.0, a[2] + a[5] / 2.0
    bz_lo, bz_hi = b[2] - b[5] / 2.0, b[2] + b[5] / 2.0
    lo = max(az_lo, bz_lo)
    hi = min(az_hi, bz_hi)
    return max(hi - lo, 0.0)


def _iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    bev = _bev_iou(a, b)
    if bev == 0.0:
        return 0.0
    z = _z_overlap(a, b)
    if z == 0.0:
        return 0.0
    pa = Polygon(_box_corners_2d(a))
    pb = Polygon(_box_corners_2d(b))
    inter_bev = pa.intersection(pb).area
    inter_3d = inter_bev * z
    vol_a = a[3] * a[4] * a[5]
    vol_b = b[3] * b[4] * b[5]
    return inter_3d / (vol_a + vol_b - inter_3d) if (vol_a + vol_b - inter_3d) > 0 else 0.0


def _compute_ap_11(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """KITTI-style 11-point interpolated AP (Pascal VOC 2007)."""
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        prec_at = precisions[recalls >= r]
        ap += prec_at.max() if prec_at.size else 0.0
    return ap / 11.0


def _compute_ap_40(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """KITTI 40-point interpolated AP (used since 2019)."""
    ap = 0.0
    for r in np.linspace(1.0 / 40.0, 1.0, 40):
        prec_at = precisions[recalls >= r]
        ap += prec_at.max() if prec_at.size else 0.0
    return ap / 40.0


def _ap_for_class(
    predictions: dict, ground_truth: dict, cls: str,
    iou_thresholds: tuple, iou_fn,
) -> dict:
    """AP for a single class at each IoU threshold; returns a dict per
    threshold with R11 and R40 numbers + counts."""
    n_gt_total = sum(
        sum(1 for g in gts if str(g.get("class", "")).lower() == cls.lower())
        for gts in ground_truth.values()
    )
    out = {"n_gt": n_gt_total, "n_pred": 0}

    if n_gt_total == 0:
        for thr in iou_thresholds:
            out[f"AP_{thr}_R11"] = float("nan")  # excluded from mean
            out[f"AP_{thr}_R40"] = float("nan")
        return out

    n_pred_total = 0
    for thr in iou_thresholds:
        all_scores, all_tp, all_fp = [], [], []
        for frame_id, gts in ground_truth.items():
            class_gts = [g for g in gts if str(g.get("class", "")).lower() == cls.lower()]
            if not class_gts:
                # frames with no GT of this class still contribute false positives
                preds = [
                    p for p in predictions.get(frame_id, [])
                    if str(p.get("class", "")).lower() == cls.lower()
                ]
                for p in preds:
                    all_scores.append(p["score"])
                    all_tp.append(0)
                    all_fp.append(1)
                continue
            preds = [
                p for p in predictions.get(frame_id, [])
                if str(p.get("class", "")).lower() == cls.lower()
            ]
            preds = sorted(preds, key=lambda d: -d["score"])
            gt_boxes = [np.asarray(g["box"]) for g in class_gts]
            gt_used = [False] * len(gt_boxes)
            for p in preds:
                pbox = np.asarray(p["box"])
                best_iou, best_j = 0.0, -1
                for j, gtb in enumerate(gt_boxes):
                    if gt_used[j]:
                        continue
                    iou = iou_fn(pbox, gtb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                all_scores.append(p["score"])
                if best_iou >= thr and best_j >= 0:
                    all_tp.append(1); all_fp.append(0)
                    gt_used[best_j] = True
                else:
                    all_tp.append(0); all_fp.append(1)
        n_pred_total = len(all_scores)

        if not all_scores:
            out[f"AP_{thr}_R11"] = 0.0
            out[f"AP_{thr}_R40"] = 0.0
            continue
        order = np.argsort(-np.asarray(all_scores))
        tp = np.cumsum(np.asarray(all_tp)[order])
        fp = np.cumsum(np.asarray(all_fp)[order])
        precisions = tp / np.maximum(tp + fp, 1)
        recalls = tp / n_gt_total
        out[f"AP_{thr}_R11"] = float(_compute_ap_11(precisions, recalls))
        out[f"AP_{thr}_R40"] = float(_compute_ap_40(precisions, recalls))

    out["n_pred"] = n_pred_total
    return out


def evaluate_3d_detection(
    predictions: dict,
    ground_truth: dict,
    class_names: list,
    iou_thresholds=(0.5, 0.7),
    iou_type: str = "3d",
) -> dict:
    """Compute per-class AP over all frames, then mean AP across classes
    (NaN-skipping; classes with no GT in the held-out set are excluded
    from the mean).

    predictions: {frame_id: list of {"box": [7], "score": float, "class": str}}
    ground_truth: {frame_id: list of {"box": [7], "class": str}}
    class_names: list of valid class strings for this track
    iou_type: "3d" or "bev"

    Returns: flat dict of mean APs and per-class APs, e.g.
        {
          "mAP_0.5_R40": float, "mAP_0.7_R40": float,
          "mAP_0.5_R11": float, "mAP_0.7_R11": float,
          "per_class": {
            "Car":        {"AP_0.5_R40": ..., "n_gt": ..., "n_pred": ...},
            "Pedestrian": {...},
            ...
          },
          "n_gt": int, "n_pred": int,
        }
    """
    iou_fn = _iou_3d if iou_type == "3d" else _bev_iou
    per_class = {}
    for cls in class_names:
        per_class[cls] = _ap_for_class(
            predictions, ground_truth, cls, iou_thresholds, iou_fn
        )

    out = {"per_class": per_class}
    out["n_gt"] = sum(c["n_gt"] for c in per_class.values())
    out["n_pred"] = sum(c["n_pred"] for c in per_class.values())

    for thr in iou_thresholds:
        for tag in ("R11", "R40"):
            vals = [c[f"AP_{thr}_{tag}"] for c in per_class.values()
                    if not np.isnan(c[f"AP_{thr}_{tag}"])]
            out[f"mAP_{thr}_{tag}"] = float(np.mean(vals)) if vals else 0.0

    return out
