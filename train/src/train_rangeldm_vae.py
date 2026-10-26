#!/usr/bin/env python3
"""Train the RangeLDM-S2R range/mask autoencoder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class RangeVAEDataset(Dataset):
    def __init__(self, root: Path, width: int) -> None:
        self.files = sorted((root / "synthetic").glob("*.npz"))
        self.files += sorted((root / "real").glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(root)
        self.width = width

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        with np.load(self.files[index]) as data:
            log_range = torch.from_numpy(data["log_range"].copy()).float()[None]
            mask = torch.from_numpy(data["mask"].copy()).float()[None]
            if "semantic" in data:
                obj = torch.from_numpy((data["semantic"].sum(-1) > 0).astype(np.float32))[None]
                boundary = torch.from_numpy(data["object_boundary"].copy()).float()[None]
            else:
                obj, boundary = torch.zeros_like(mask), torch.zeros_like(mask)
        depth = torch.clamp(log_range, 0, 1) * 2 - 1
        image = torch.cat([depth, mask], 0)
        if image.shape[-1] != self.width:
            image = F.interpolate(image[None], (image.shape[-2], self.width), mode="bilinear", align_corners=False)[0]
            obj = F.interpolate(obj[None], (obj.shape[-2], self.width), mode="nearest")[0]
            boundary = F.interpolate(boundary[None], (boundary.shape[-2], self.width), mode="nearest")[0]
        return image, obj, boundary


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/rangeldm_s2r_2k"))
    p.add_argument("--output", type=Path, default=Path("outputs/rangeldm_vae_m1"))
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epoch-offset", type=int, default=0)
    p.add_argument("--object-weight", type=float, default=8.0)
    p.add_argument("--boundary-weight", type=float, default=4.0)
    p.add_argument("--levels", type=int, choices=[3, 4], default=4)
    p.add_argument("--seed", type=int, default=20260724)
    return p.parse_args()


def edge_map(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    return dx + dy


def main() -> None:
    cfg = args()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    torch.manual_seed(cfg.seed + rank)

    dataset = RangeVAEDataset(cfg.data, cfg.width)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=cfg.workers, pin_memory=True,
        drop_last=True, persistent_workers=cfg.workers > 0
    )
    if cfg.resume:
        model = AutoencoderKL.from_pretrained(cfg.resume).to(device)
    else:
        channels = (64, 128, 256) if cfg.levels == 3 else (32, 64, 128, 128)
        model = AutoencoderKL(
        in_channels=2, out_channels=2, latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * cfg.levels,
        up_block_types=("UpDecoderBlock2D",) * cfg.levels,
        block_out_channels=channels,
        layers_per_block=2, norm_num_groups=8,
        sample_size=(128, cfg.width), scaling_factor=0.18215,
        ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    cfg.output.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        running = torch.zeros(5, device=device)
        model.train()
        for image, obj, boundary in loader:
            image = image.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                posterior = model.module.encode(image).latent_dist if distributed else model.encode(image).latent_dist
                latent = posterior.sample()
                reconstruction = model.module.decode(latent).sample if distributed else model.decode(latent).sample
                valid = image[:, 1:2]
                depth_weight = valid * (1 + cfg.object_weight * obj + cfg.boundary_weight * boundary)
                depth_loss = (torch.abs(reconstruction[:, :1] - image[:, :1]) * depth_weight).sum() / depth_weight.sum().clamp_min(1)
                mask_loss = F.binary_cross_entropy_with_logits(reconstruction[:, 1:2], valid, weight=1 + cfg.object_weight * obj)
                edge_weight = 1 + cfg.boundary_weight * boundary
                edge_loss = (torch.abs(edge_map(reconstruction[:, :1]) - edge_map(image[:, :1])) * edge_weight).sum() / edge_weight.sum().clamp_min(1)
                kl_loss = posterior.kl().mean()
                loss = depth_loss + 0.5 * mask_loss + 0.25 * edge_loss + 1e-6 * kl_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += torch.tensor(
                [loss.item(), depth_loss.item(), mask_loss.item(), edge_loss.item(), 1],
                device=device
            )
        if distributed:
            dist.all_reduce(running)
        metrics = (running[:4] / running[4]).cpu().tolist()
        actual_epoch = epoch + cfg.epoch_offset
        record = dict(epoch=actual_epoch, loss=metrics[0], depth=metrics[1], mask=metrics[2], edge=metrics[3])
        history.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            (cfg.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                raw = model.module if distributed else model
                raw.save_pretrained(cfg.output / f"epoch_{actual_epoch:03d}", safe_serialization=True)
                torch.save(optimizer.state_dict(), cfg.output / f"optimizer_{actual_epoch:03d}.pt")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
