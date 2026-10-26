#!/usr/bin/env python3
"""SDEdit-style synthetic-to-real translation in the learned LiDAR latent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/rangeldm_s2r_m0/synthetic"))
    p.add_argument("--vae", type=Path, default=Path("outputs/rangeldm_vae_m1_v2/epoch_020"))
    p.add_argument("--unet", type=Path, default=Path("outputs/rangeldm_real_ldm_m2_v2/epoch_100"))
    p.add_argument("--scheduler", type=Path, default=Path("outputs/rangeldm_real_ldm_m2_v2/scheduler"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, default=Path("outputs/sensor_profile_v2"))
    p.add_argument("--strengths", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--anchor-objects", action="store_true")
    return p.parse_args()


def load_batch(paths: list[Path], width: int):
    images, source_range, source_mask, source_xyz, semantic = [], [], [], [], []
    for path in paths:
        with np.load(path) as d:
            log_range = torch.from_numpy(d["log_range"].copy()).float()[None]
            mask = torch.from_numpy(d["mask"].copy()).float()[None]
            source_range.append(d["range"].copy())
            source_mask.append(d["mask"].copy())
            source_xyz.append(d["xyz"].copy())
            semantic.append(d["semantic"].copy())
        depth = torch.clamp(log_range, 0, 1) * 2 - 1
        image = torch.cat([depth, mask], 0)
        image = F.interpolate(image[None], (128, width), mode="bilinear", align_corners=False)[0]
        images.append(image)
    return torch.stack(images), source_range, source_mask, source_xyz, semantic


def reconstruct(decoded: torch.Tensor, ring_angles: np.ndarray):
    decoded = F.interpolate(decoded, (128, 1024), mode="bilinear", align_corners=False)
    scaled_log = torch.clamp((decoded[:, 0] + 1) / 2, 0, 1)
    ranges = torch.pow(2.0, scaled_log * 8.0) - 1.0
    masks = torch.sigmoid(decoded[:, 1]) >= 0.5
    ranges = ranges * masks
    r = ranges.cpu().numpy()
    m = masks.cpu().numpy()
    az = ((np.arange(1024) + 0.5) / 1024 * 2 * np.pi - np.pi)[None, :]
    el = ring_angles[:, None]
    dirs = np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
         np.broadcast_to(np.sin(el), (128, 1024))], axis=-1
    ).astype(np.float32)
    xyz = r[..., None] * dirs[None]
    return r, m, xyz


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    paths = sorted(cfg.data.glob("*.npz"))[cfg.start:cfg.start + cfg.frames]
    if not paths:
        raise FileNotFoundError(cfg.data)
    vae = AutoencoderKL.from_pretrained(cfg.vae).to(device).eval()
    unet = UNet2DModel.from_pretrained(cfg.unet).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(cfg.scheduler)
    ring_angles = np.load(cfg.profile / "ring_angles.npy").astype(np.float32)
    cfg.output.mkdir(parents=True, exist_ok=True)
    reports = []

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for strength in cfg.strengths:
            out_dir = cfg.output / f"strength_{strength:.2f}"
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            generator = torch.Generator(device=device).manual_seed(cfg.seed + round(strength * 1000))
            for start in range(0, len(paths), cfg.batch_size):
                subset = paths[start:start + cfg.batch_size]
                image, src_ranges, src_masks, src_xyz, semantics = load_batch(subset, cfg.width)
                image = image.to(device)
                latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor
                scheduler.set_timesteps(cfg.steps, device=device)
                init_steps = max(1, min(cfg.steps, int(round(cfg.steps * strength))))
                timesteps = scheduler.timesteps[cfg.steps - init_steps:]
                noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
                t0 = timesteps[0].expand(latent.shape[0])
                sample = scheduler.add_noise(latent, noise, t0)
                for timestep in timesteps:
                    prediction = unet(sample, timestep).sample
                    sample = scheduler.step(prediction, timestep, sample).prev_sample
                decoded = vae.decode(sample / vae.config.scaling_factor).sample.float()
                ranges, masks, xyz = reconstruct(decoded, ring_angles)
                for i, path in enumerate(subset):
                    src_r = src_ranges[i]
                    src_m = source_valid = src_masks[i] > 0
                    obj = semantics[i].sum(-1) > 0
                    if cfg.anchor_objects:
                        ranges[i][obj] = src_r[obj]
                        masks[i][obj] = source_valid[obj]
                        xyz[i][obj] = src_xyz[i][obj]
                    common = source_valid & masks[i]
                    object_common = common & obj
                    row = {
                        "frame": path.stem,
                        "source_points": int(source_valid.sum()),
                        "generated_points": int(masks[i].sum()),
                        "mask_iou": float((source_valid & masks[i]).sum() / max((source_valid | masks[i]).sum(), 1)),
                        "depth_mae_common_m": float(np.abs(ranges[i][common] - src_r[common]).mean()) if common.any() else None,
                        "object_depth_mae_m": float(np.abs(ranges[i][object_common] - src_r[object_common]).mean()) if object_common.any() else None,
                        "object_ray_recall": float(object_common.sum() / max(obj.sum(), 1)),
                    }
                    rows.append(row)
                    points = xyz[i][masks[i]]
                    np.save(out_dir / f"{path.stem}.npy", points.astype(np.float32))
            valid_depth = [r["depth_mae_common_m"] for r in rows if r["depth_mae_common_m"] is not None]
            valid_object = [r["object_depth_mae_m"] for r in rows if r["object_depth_mae_m"] is not None]
            summary = {
                "strength": strength,
                "frames": len(rows),
                "generated_points_median": float(np.median([r["generated_points"] for r in rows])),
                "mask_iou_median": float(np.median([r["mask_iou"] for r in rows])),
                "depth_mae_common_m_median": float(np.median(valid_depth)),
                "object_depth_mae_m_median": float(np.median(valid_object)),
                "object_ray_recall_median": float(np.median([r["object_ray_recall"] for r in rows])),
            }
            (out_dir / "audit.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n")
            reports.append(summary)
            print(json.dumps(summary), flush=True)
    (cfg.output / "strength_scan.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
