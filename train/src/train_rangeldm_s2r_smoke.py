#!/usr/bin/env python3
"""Smoke-test the UrbanTwin -> V2X-Real conditional diffusion data path.

This deliberately small pixel-space model is an integration test.  The full
experiment replaces it with the RangeLDM VAE latent and larger conditional
UNet after data normalization, conditioning, loss, and checkpointing pass.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, Dataset


class UnpairedRangeDataset(Dataset):
    def __init__(self, root: Path, width: int = 256) -> None:
        self.synthetic = sorted((root / "synthetic").glob("*.npz"))
        self.real = sorted((root / "real").glob("*.npz"))
        if not self.synthetic or not self.real:
            raise FileNotFoundError(f"missing synthetic/real npz files below {root}")
        self.width = width

    def __len__(self) -> int:
        return max(len(self.synthetic), len(self.real))

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}

    @staticmethod
    def _resize(x: torch.Tensor, width: int, mode: str = "bilinear") -> torch.Tensor:
        if x.shape[-1] == width:
            return x
        kwargs = {"size": (x.shape[-2], width), "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(x[None], **kwargs)[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        syn = self._load(self.synthetic[index % len(self.synthetic)])
        # Deterministic coprime permutation makes pairing explicitly unpaired.
        real = self._load(self.real[(index * 37 + 11) % len(self.real)])

        def chw(x: np.ndarray) -> torch.Tensor:
            if x.ndim == 2:
                x = x[None]
            elif x.ndim == 3:
                x = np.moveaxis(x, -1, 0)
            return torch.from_numpy(x.astype(np.float32, copy=False))

        syn_mask = chw(syn["mask"])
        real_mask = chw(real["mask"])
        syn_log = chw(syn["log_range"])
        real_log = chw(real["log_range"])
        # log1p(120 m) ~= 4.80. Map depth to [-1, 1], preserve empty rays.
        syn_depth = torch.clamp(syn_log, 0, 1) * 2 - 1
        real_depth = torch.clamp(real_log, 0, 1) * 2 - 1
        normal = chw(syn["normal"])
        edge = chw(syn["edge"])
        semantic = chw(syn["semantic"])
        boundary = chw(syn["object_boundary"])
        condition = torch.cat(
            [syn_depth, syn_mask, normal, edge, semantic, boundary], dim=0
        )
        target = torch.cat([real_depth, real_mask * 2 - 1], dim=0)
        condition = self._resize(condition, self.width)
        target = self._resize(target, self.width)
        return {"condition": condition, "target": target}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/rangeldm_s2r_m0"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rangeldm_s2r_smoke"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    dataset = UnpairedRangeDataset(args.data, args.width)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True
    )
    iterator = iter(loader)
    condition_channels, target_channels = 10, 2
    model = UNet2DModel(
        sample_size=(128, args.width),
        in_channels=condition_channels + target_channels,
        out_channels=target_channels,
        layers_per_block=1,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    ).cuda()
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon", clip_sample=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    losses: list[float] = []

    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        condition = batch["condition"].cuda(non_blocking=True)
        target = batch["target"].cuda(non_blocking=True)
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (target.shape[0],),
            device=target.device, dtype=torch.long
        )
        noisy = scheduler.add_noise(target, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            prediction = model(torch.cat([noisy, condition], 1), timesteps).sample
            loss = F.mse_loss(prediction.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss))
        if step == 1 or step % 10 == 0:
            print(f"step={step:04d} loss={losses[-1]:.6f} grad={float(grad_norm):.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
        args.output / "smoke_last.pt",
    )
    report = {
        "steps": args.steps,
        "samples": len(dataset),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_10": float(np.mean(losses[-10:])),
        "finite": bool(np.isfinite(losses).all()),
        "checkpoint": str(args.output / "smoke_last.pt"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
