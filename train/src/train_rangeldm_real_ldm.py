#!/usr/bin/env python3
"""Train a real-domain latent diffusion prior for RangeLDM-S2R."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class RealRangeDataset(Dataset):
    def __init__(self, root: Path, width: int) -> None:
        self.files = sorted((root / "real").glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(root / "real")
        self.width = width

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with np.load(self.files[index]) as data:
            depth = torch.from_numpy(data["log_range"].copy()).float()[None]
            mask = torch.from_numpy(data["mask"].copy()).float()[None]
        depth = torch.clamp(depth, 0, 1) * 2 - 1
        image = torch.cat([depth, mask], 0)
        if image.shape[-1] != self.width:
            image = F.interpolate(
                image[None], (image.shape[-2], self.width),
                mode="bilinear", align_corners=False
            )[0]
        return image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/rangeldm_s2r_2k"))
    p.add_argument("--vae", type=Path, default=Path("outputs/rangeldm_vae_m1/epoch_020"))
    p.add_argument("--output", type=Path, default=Path("outputs/rangeldm_real_ldm_m2"))
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260724)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    rank = dist.get_rank() if distributed else 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(cfg.seed + rank)

    dataset = RealRangeDataset(cfg.data, cfg.width)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=cfg.workers, pin_memory=True,
        drop_last=True, persistent_workers=cfg.workers > 0
    )
    vae = AutoencoderKL.from_pretrained(cfg.vae).to(device).eval()
    vae.requires_grad_(False)
    latent_h, latent_w = 16, cfg.width // 8
    model = UNet2DModel(
        sample_size=(latent_h, latent_w), in_channels=4, out_channels=4,
        layers_per_block=2, block_out_channels=(128, 256, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon", clip_sample=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    cfg.output.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        running = torch.zeros(2, device=device)
        for image in loader:
            image = image.to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                latent = vae.encode(image).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
            noise = torch.randn_like(latent)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latent.shape[0],), device=device, dtype=torch.long
            )
            noisy = noise_scheduler.add_noise(latent, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                prediction = model(noisy, timesteps).sample
                loss = F.mse_loss(prediction.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += torch.tensor([loss.item(), 1], device=device)
        if distributed:
            dist.all_reduce(running)
        record = {"epoch": epoch, "loss": float((running[0] / running[1]).cpu())}
        history.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            (cfg.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                raw = model.module if distributed else model
                raw.save_pretrained(cfg.output / f"epoch_{epoch:03d}", safe_serialization=True)
                noise_scheduler.save_pretrained(cfg.output / "scheduler")
                torch.save(optimizer.state_dict(), cfg.output / f"optimizer_{epoch:03d}.pt")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
