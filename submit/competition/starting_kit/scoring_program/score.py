#!/usr/bin/env python3
"""
Codabench scoring entrypoint for the UrbanTwin sim2real LiDAR challenge.

Container layout (set by the Codabench worker):
    /app/program        this scoring program
    /app/input/ref      reference data bundled into the task
    /app/input/res      participant submission, extracted
    /app/output         where we write scores.json + detailed_results.html

Submission expected at /app/input/res/:
    synthetic/<frame_id>.bin     50 LiDAR frames, float32 N x 3 (x,y,z)
                                 max 125,000 points per frame
    predictions.json             detections on the held-out real test frames
    declaration.txt              signed honor declaration

Reference at /app/input/ref/:
    config.json                  track metadata: name, pc_range, expected ids
    realism_reference/<id>.bin   50 real LiDAR frames (held out, organizer only)
    detection_test_frame_ids.json  expected frame IDs for predictions
    detection_gt.json            ground-truth boxes for those frames
    forbidden_frames.txt         do-not-train-on list (informational here)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

import metrics
from detection_eval import evaluate_3d_detection


REF_DIR = Path(os.environ.get("CODABENCH_REF", "/app/input/ref"))
RES_DIR = Path(os.environ.get("CODABENCH_RES", "/app/input/res"))
OUT_DIR = Path(os.environ.get("CODABENCH_OUT", "/app/output"))


def load_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)
    if raw.size % 4 == 0:
        # Tolerate (x,y,z,intensity); drop intensity.
        return raw.reshape(-1, 4)[:, :3]
    raise ValueError(
        f"{path.name}: {raw.size} float32s is not divisible by 3 (xyz) or 4 (xyzi)"
    )


def fail(msg: str) -> None:
    """Write a zero-score result and abort with explanation rather than
    crashing the worker uninformatively."""
    print(f"[scoring] FATAL: {msg}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = {
        "score_combined": 0.0,
        "score_detection": 0.0,
        "score_realism": 0.0,
        "mAP_3d_0.5_R40": 0.0,
        "mAP_3d_0.7_R40": 0.0,
        "mAP_bev_0.5_R40": 0.0,
        "CD": float("inf"),
        "MMD": float("inf"),
        "EMD": float("inf"),
        "FPD": float("inf"),
        "valid_submission": 0,
        "error_message": msg,
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(scores, indent=2))
    (OUT_DIR / "detailed_results.html").write_text(
        f"<h2>Submission rejected</h2><pre>{msg}</pre>"
    )
    sys.exit(0)  # do NOT non-zero-exit; Codabench reports the score row instead


def _has_valid_declaration() -> bool:
    """Accept either a signed PDF (existence check; signature not verified)
    or a plain text declaration containing the required statement."""
    pdf = RES_DIR / "declaration.pdf"
    txt = RES_DIR / "declaration.txt"
    if pdf.exists() and pdf.stat().st_size > 100:
        # quick magic check
        head = pdf.read_bytes()[:5]
        if head[:4] == b"%PDF":
            return True
    if txt.exists():
        decl = txt.read_text(encoding="utf-8", errors="replace")
        required_phrase = "I confirm that I did not train on any frame listed in"
        if required_phrase.lower() in decl.lower():
            return True
    return False


def validate_submission(config: dict) -> tuple[list, dict]:
    if not _has_valid_declaration():
        fail(
            "Honor declaration is missing or invalid. Submit either "
            "declaration.pdf (signed) or declaration.txt containing the required "
            "statement (see submission_format)."
        )

    syn_dir = RES_DIR / "synthetic"
    if not syn_dir.is_dir():
        fail("Submission must contain a 'synthetic/' directory with 50 .bin files.")
    syn_files = sorted(syn_dir.glob("*.bin"))
    expected_n = config["n_realism_frames"]
    if len(syn_files) != expected_n:
        fail(f"Expected {expected_n} .bin files in synthetic/, got {len(syn_files)}.")
    for f in syn_files:
        if f.stat().st_size > 1_500_000:
            fail(f"{f.name} exceeds 1.5MB cap (~125,000 points * 12 bytes).")

    pred_path = RES_DIR / "predictions.json"
    if not pred_path.exists():
        fail("predictions.json is missing from the submission.")
    try:
        preds = json.loads(pred_path.read_text())
    except Exception as e:  # noqa: BLE001
        fail(f"predictions.json is not valid JSON: {e}")

    expected_ids = set(config["detection_test_frame_ids"])
    if set(preds.keys()) != expected_ids:
        missing = expected_ids - set(preds.keys())
        extra = set(preds.keys()) - expected_ids
        fail(
            f"predictions.json frame IDs do not match expected. "
            f"missing={sorted(list(missing))[:5]} extra={sorted(list(extra))[:5]}"
        )

    valid_classes = set(config.get("class_names", []))
    for fid, dets in preds.items():
        if not isinstance(dets, list):
            fail(f"predictions[{fid}] must be a list of detections.")
        for k, det in enumerate(dets):
            if not (isinstance(det, dict) and "box" in det and "score" in det):
                fail(f"predictions[{fid}][{k}] missing 'box' or 'score'.")
            if len(det["box"]) != 7:
                fail(f"predictions[{fid}][{k}].box must have 7 values, got {len(det['box'])}.")
            # `class` is optional but recommended; default to "Car"
            det.setdefault("class", "Car")

    return syn_files, preds


def normalize(value: float, lo: float, hi: float, lower_is_better: bool) -> float:
    """Min-max into [0, 1]; clip outside."""
    if not np.isfinite(value):
        return 0.0
    v = max(min(value, hi), lo)
    score = (v - lo) / (hi - lo) if hi > lo else 0.0
    return 1.0 - score if lower_is_better else score


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = REF_DIR / "config.json"
    if not config_path.exists():
        fail(f"Reference data is missing config.json (looked at {config_path}).")
    config = json.loads(config_path.read_text())

    syn_files, preds = validate_submission(config)

    pc_range = tuple(config["point_cloud_range"])

    real_ref_dir = REF_DIR / "realism_reference"
    real_files = sorted(real_ref_dir.glob("*.bin"))
    if len(real_files) != len(syn_files):
        fail(
            f"Reference data has {len(real_files)} real frames but submission "
            f"has {len(syn_files)} synthetic. Track config drift."
        )

    syn_clouds = [load_bin(f) for f in syn_files]
    real_clouds = [load_bin(f) for f in real_files]

    print("[scoring] computing realism metrics...", flush=True)
    realism = metrics.compute_realism(syn_clouds, real_clouds, pc_range, seed=1234)
    print(f"[scoring] realism: {realism}", flush=True)

    print("[scoring] computing 3D detection mAP...", flush=True)
    gt = json.loads((REF_DIR / "detection_gt.json").read_text())
    class_names = config.get("class_names", [])
    pred_for_eval = {fid: preds[fid] for fid in gt.keys()}
    det_3d = evaluate_3d_detection(
        pred_for_eval, gt, class_names,
        iou_thresholds=(0.5, 0.7), iou_type="3d",
    )
    det_bev = evaluate_3d_detection(
        pred_for_eval, gt, class_names,
        iou_thresholds=(0.5, 0.7), iou_type="bev",
    )
    print(f"[scoring] mAP_3d_0.5_R40 = {det_3d.get('mAP_0.5_R40')}", flush=True)

    norms = config["normalization"]
    realism_norm = np.mean([
        normalize(realism["CD"], norms["CD"][0], norms["CD"][1], lower_is_better=True),
        normalize(realism["MMD"], norms["MMD"][0], norms["MMD"][1], lower_is_better=True),
        normalize(realism["EMD"], norms["EMD"][0], norms["EMD"][1], lower_is_better=True),
        normalize(realism["FPD"], norms["FPD"][0], norms["FPD"][1], lower_is_better=True),
    ])
    detection_norm = normalize(
        det_3d["mAP_0.5_R40"], norms["AP"][0], norms["AP"][1], lower_is_better=False
    )
    combined = 0.6 * detection_norm + 0.4 * realism_norm

    scores = {
        "score_combined": round(float(combined), 6),
        "score_detection": round(float(detection_norm), 6),
        "score_realism": round(float(realism_norm), 6),
        "mAP_3d_0.5_R40": round(float(det_3d["mAP_0.5_R40"]), 6),
        "mAP_3d_0.5_R11": round(float(det_3d["mAP_0.5_R11"]), 6),
        "mAP_3d_0.7_R40": round(float(det_3d["mAP_0.7_R40"]), 6),
        "mAP_bev_0.5_R40": round(float(det_bev["mAP_0.5_R40"]), 6),
        "CD": round(float(realism["CD"]), 6),
        "MMD": round(float(realism["MMD"]), 8),
        "EMD": round(float(realism["EMD"]), 6),
        "FPD": round(float(realism["FPD"]), 6),
        "valid_submission": 1,
    }
    # Per-class APs (also written, but only mAP* show on the leaderboard)
    for cls, info in det_3d["per_class"].items():
        ap = info.get("AP_0.5_R40", float("nan"))
        scores[f"AP_3d_0.5_R40_{cls}"] = (
            round(float(ap), 6) if (isinstance(ap, float) and not np.isnan(ap)) else None
        )

    (OUT_DIR / "scores.json").write_text(json.dumps(scores, indent=2))
    write_detailed_results(scores, det_3d, det_bev, realism, config)
    print(f"[scoring] DONE: scores={scores}", flush=True)


def _format_class_row(cls: str, info: dict) -> str:
    ap = info.get("AP_0.5_R40", float("nan"))
    if isinstance(ap, float) and ap != ap:  # NaN
        ap_txt = "n/a (no GT)"
    else:
        ap_txt = f"{ap:.4f}"
    return (
        f"<tr><td>{cls}</td><td>{ap_txt}</td>"
        f"<td>{info['n_gt']}</td><td>{info['n_pred']}</td></tr>"
    )


def write_detailed_results(scores, det_3d, det_bev, realism, config) -> None:
    track = config.get("track_name", "track")
    html = [
        f"<h1>UrbanTwin sim2real challenge — {track}</h1>",
        "<h2>Combined</h2>",
        f"<p><b>Combined score:</b> {scores['score_combined']:.4f} "
        f"(0.6 * detection + 0.4 * realism)</p>",
        f"<p>Detection (normalized): {scores['score_detection']:.4f}</p>",
        f"<p>Realism (normalized): {scores['score_realism']:.4f}</p>",
        "<h2>3D Object Detection &mdash; mean AP across all categories</h2>",
        "<table border=1 cellpadding=4><tr><th></th><th>3D mAP</th><th>BEV mAP</th></tr>",
        f"<tr><td>IoU 0.5 (R40)</td><td>{det_3d['mAP_0.5_R40']:.4f}</td>"
        f"<td>{det_bev['mAP_0.5_R40']:.4f}</td></tr>",
        f"<tr><td>IoU 0.7 (R40)</td><td>{det_3d['mAP_0.7_R40']:.4f}</td>"
        f"<td>{det_bev['mAP_0.7_R40']:.4f}</td></tr>",
        f"<tr><td>IoU 0.5 (R11)</td><td>{det_3d['mAP_0.5_R11']:.4f}</td>"
        f"<td>{det_bev['mAP_0.5_R11']:.4f}</td></tr>",
        "</table>",
        f"<p>Predictions: {det_3d['n_pred']} on {det_3d['n_gt']} ground-truth boxes.</p>",
        "<h3>Per-class 3D AP @ IoU 0.5 (R40)</h3>",
        "<table border=1 cellpadding=4>",
        "<tr><th>Class</th><th>AP</th><th>n_gt (held-out 50)</th><th>n_pred</th></tr>",
        *(_format_class_row(cls, info) for cls, info in det_3d["per_class"].items()),
        "</table>",
        "<h2>Distributional realism (raw-point space, 10,000 subsample, seed 1234)</h2>",
        "<table border=1 cellpadding=4>",
        "<tr><th>Metric</th><th>Value</th><th>Lower is better?</th></tr>",
        f"<tr><td>Chamfer Distance (CD)</td><td>{realism['CD']:.4f}</td><td>yes</td></tr>",
        f"<tr><td>Maximum Mean Discrepancy (MMD, RBF, median heuristic)</td>"
        f"<td>{realism['MMD']:.4e}</td><td>yes</td></tr>",
        f"<tr><td>Earth Mover's Distance (EMD)</td><td>{realism['EMD']:.4f}</td><td>yes</td></tr>",
        f"<tr><td>Fr&eacute;chet Point-cloud Distance (FPD)</td>"
        f"<td>{realism['FPD']:.4f}</td><td>yes</td></tr>",
        "</table>",
        f"<p>Pre-subsample point counts &mdash; synthetic: "
        f"{realism['n_syn_points_pre_subsample']}, real reference: "
        f"{realism['n_real_points_pre_subsample']}. "
        f"MMD bandwidth gamma: {realism['MMD_gamma']:.4e}.</p>",
        "<p>Definitions and protocol: see the Evaluation tab.</p>",
    ]
    (OUT_DIR / "detailed_results.html").write_text("\n".join(html))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail("Unhandled exception in scoring program:\n" + traceback.format_exc())
