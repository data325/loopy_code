#!/usr/bin/env python3
"""Export LiDM range-image reconstructions as native M1 point clouds.

The M1 corpus used by the LiDM experiment preserves 64x1024 LiDAR ordering.
This exporter uses the per-ring elevations estimated from the original scans,
instead of assuming uniformly spaced elevation angles at export time.  It is
therefore suitable for an honest offline realism comparison; it does not read
or use any hidden evaluation data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


PROJECT = Path(__file__).resolve().parents[1]
LIDM = PROJECT / "third_party" / "lidar_diffusion_cvpr24"
# The official repository is checked out under third_party, while its two
# editable dependencies are intentionally shared at the project-level src/.
sys.path[:0] = [str(LIDM), str(PROJECT / "src" / "taming-transformers"), str(PROJECT / "src" / "clip")]

from lidm.utils.misc_utils import instantiate_from_config  # noqa: E402


PC_MIN = np.asarray([-40.0, -40.0, -8.0], dtype=np.float32)
PC_MAX = np.asarray([40.0, 40.0, 2.0], dtype=np.float32)


def native_range_to_xyz(log_range: np.ndarray, mask: np.ndarray, ring_angles: np.ndarray) -> np.ndarray:
    """Convert log2(range + 1) / 8 on M1's organized grid to XYZ."""
    height, width = log_range.shape
    if ring_angles.shape != (height,):
        raise ValueError(f"ring angles {ring_angles.shape} do not match grid height {height}")
    depth = np.exp2(np.clip(log_range, 0.0, 1.0) * 8.0) - 1.0
    yaw = np.pi * (2.0 * np.arange(width, dtype=np.float32) / width - 1.0)
    pitch = ring_angles.astype(np.float32, copy=False)
    cp, sp = np.cos(pitch)[:, None], np.sin(pitch)[:, None]
    cy, sy = np.cos(yaw)[None, :], np.sin(yaw)[None, :]
    xyz = np.stack((cp * cy * depth, -cp * sy * depth, sp * depth), axis=-1)
    valid = mask & np.isfinite(xyz).all(axis=-1)
    valid &= (xyz >= PC_MIN).all(axis=-1) & (xyz <= PC_MAX).all(axis=-1)
    return xyz[valid].astype(np.float32, copy=False)


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    config = OmegaConf.load(config_path)
    # The perceptual/discriminator loss is unnecessary for inference and can
    # bring optional training-only dependencies into an evaluation process.
    if "lossconfig" in config.model.params:
        del config.model.params.lossconfig
    model = instantiate_from_config(config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if not key.startswith("loss.")]
    bad_unexpected = [key for key in unexpected if not key.startswith("loss.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--oracle-mask", action="store_true", help="Audit only: use source mask, never for generation/submission")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--range-calibrator", type=Path, default=None, help="Training-split fitted range LUT; applied blindly at export")
    args = parser.parse_args()

    paths = sorted((args.corpus / args.split).glob("*.npz"))
    if not paths:
        raise SystemExit(f"No {args.split} scans in {args.corpus}")
    ring_angles = np.load(args.corpus / "ring_angles.npy").astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.config, args.checkpoint, device)
    calibrator = None
    if args.range_calibrator is not None:
        calibrator = np.load(args.range_calibrator)
        if "knots" not in calibrator or "values" not in calibrator:
            raise ValueError("range calibrator must contain knots and values")
        if calibrator["knots"].shape != (64, calibrator["knots"].shape[1]):
            raise ValueError(f"unexpected calibrator shape {calibrator['knots'].shape}")

    total_depth_error = 0.0
    total_depth_values = 0
    true_positive = false_positive = false_negative = 0
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            records = [np.load(path) for path in batch_paths]
            try:
                ranges = np.stack([record["log_range"].astype(np.float32) for record in records])
                actual_mask = np.stack([record["mask"].astype(bool) for record in records])
            finally:
                for record in records:
                    record.close()
            range_input = ranges * 2.0 - 1.0
            if model.encoder.conv_in.in_channels == 2:
                # Reconstruction audit: the two-channel encoder receives the
                # observed return mask exactly as it did during training.
                model_input = np.stack((range_input, actual_mask.astype(np.float32) * 2.0 - 1.0), axis=1)
            else:
                model_input = range_input[:, None]
            image = torch.from_numpy(model_input).to(device)
            reconstruction, _ = model(image)
            reconstruction = reconstruction.float().cpu().numpy()
            predicted_log = np.clip((reconstruction[:, 0] + 1.0) * 0.5, 0.0, 1.0)
            predicted_mask = actual_mask if args.oracle_mask else reconstruction[:, 1] < args.mask_threshold
            if calibrator is not None:
                corrected = np.empty_like(predicted_log)
                for ring in range(predicted_log.shape[1]):
                    corrected[:, ring, :] = np.interp(
                        predicted_log[:, ring, :], calibrator["knots"][ring], calibrator["values"][ring]
                    )
                predicted_log = np.clip(corrected, 0.0, 1.0)
            for path, source_range, source_mask, pred_range, pred_mask in zip(
                batch_paths, ranges, actual_mask, predicted_log, predicted_mask
            ):
                both = source_mask & pred_mask
                total_depth_error += float(np.abs(source_range[both] - pred_range[both]).sum())
                total_depth_values += int(both.sum())
                true_positive += int(both.sum())
                false_positive += int((pred_mask & ~source_mask).sum())
                false_negative += int((source_mask & ~pred_mask).sum())
                xyz = native_range_to_xyz(pred_range, pred_mask, ring_angles)
                xyz.tofile(args.output / f"{path.stem}.bin")
            print(f"exported {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    summary = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "split": args.split,
        "frames": len(paths),
        "points": int(sum(path.stat().st_size // 12 for path in args.output.glob("*.bin"))),
        "log_range_mae_on_matching_returns": total_depth_error / max(total_depth_values, 1),
        "return_precision": precision,
        "return_recall": recall,
        "return_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "mask_threshold": args.mask_threshold,
        "oracle_mask_audit_only": bool(args.oracle_mask),
        "range_calibrator": str(args.range_calibrator) if args.range_calibrator else None,
        "geometry": "native M1 per-ring elevations from corpus/ring_angles.npy",
    }
    (args.output / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
