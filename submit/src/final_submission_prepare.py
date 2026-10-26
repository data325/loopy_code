#!/usr/bin/env python3
"""Prepare the best-effort Final-phase submission.

Pipeline:
  1) infer   : run a checkpoint on the fixed real-domain validation manifest
               and cache {predictions, ground_truth}.
  2) fuse    : load cached candidate predictions (+ VoxelRCNN TTA variants),
               grid-search class-wise NMS fusion on local R40 mAP, and export
               the chosen fusion applied to the official test-frame predictions.
  3) build   : assemble submission.zip with the chosen synthetic point cloud
               and a declaration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402
if os.environ.get("GBLOBS") == "1":
    sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "GBlobs" / "tools"),
                    str(PROJECT_ROOT / "third_party" / "GBlobs")]
else:
    sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "OpenPCDet" / "tools"),
                    str(PROJECT_ROOT / "third_party" / "OpenPCDet")]
sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "V2X-Real"),
                str(PROJECT_ROOT / "competition" / "starting_kit" / "scoring_program")]

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from detection_eval import evaluate_3d_detection  # noqa: E402


CLASSES = ["vehicle", "pedestrian", "truck"]


def load_manifest(path: Path):
    return json.loads(path.read_text())


def infer_on_test_frames(config: Path, checkpoint: Path, frames: Path, frame_ids: Path,
                        output: Path, device: int, gblobs: bool = False) -> None:
    """Run a checkpoint on the official challenge test frames (LiDAR coordinates)."""
    from common import read_points

    ids_obj = json.loads(Path(frame_ids).read_text(encoding="utf-8"))
    fids = ids_obj.get("frame_ids", ids_obj) if isinstance(ids_obj, dict) else ids_obj
    fids = [str(x) for x in fids]
    files = sorted([p for p in Path(frames).iterdir() if p.suffix.lower() in {".bin", ".npy"}])
    by_stem = {p.stem: p for p in files}
    if all(x in by_stem for x in fids):
        paths = [by_stem[x] for x in fids]
    elif len(files) == len(fids):
        paths = files
    else:
        raise SystemExit("test frame files do not match frame ids")

    samples = [{"id": fid, "bin": path, "width": 3} for fid, path in zip(fids, paths)]
    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(config), cfg)
    apply_profile_local(cfg)
    dataset = LocalPublicICDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, samples, logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(str(checkpoint), logger=logger, to_cpu=True)
    torch.cuda.set_device(device)
    model.cuda().eval()
    predictions = {}
    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            pred = model.forward(batch)[0][0]
            boxes = pred["pred_boxes"].cpu().numpy()
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            predictions[sample["id"]] = [
                {"box": [float(x) for x in box[:7]], "score": float(score),
                 "class": cfg.CLASS_NAMES[int(label) - 1]}
                for box, score, label in zip(boxes, scores, labels)
            ]
            if (index + 1) % 10 == 0:
                logger.info("infer-test %d/%d", index + 1, len(samples))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, indent=1), encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(predictions),
                      "boxes": sum(map(len, predictions.values()))}))


def apply_profile_local(cfg) -> None:
    """Set three-class CenterHead inference profile (same as eval_v2xreal_public)."""
    if cfg.MODEL.DENSE_HEAD.NAME == "CenterHead":
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = 0.001
    else:
        cfg.CLASS_NAMES = ["vehicle", "pedestrian", "truck"]


class LocalPublicICDataset(DatasetTemplate):
    """Copy of the public-IC dataset (points -> model batch)."""

    def __init__(self, dataset_cfg, class_names, samples, logger):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False,
            root_path=Path("."), logger=logger,
        )
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        from common import read_points
        points = read_points(sample["bin"], sample.get("width"))
        points[:, 3] = 0.0
        return self.prepare_data({"points": points, "frame_id": sample["id"]})


def ground_truth_for_sample_local(sample):
    import copy
    import numpy as np
    import yaml
    from opencood.data_utils import SUPER_CLASS_MAP
    from opencood.utils import box_utils

    INVERSE_CLASS_MAP = {
        fine_name: super_name
        for super_name, fine_names in SUPER_CLASS_MAP.items()
        for fine_name in fine_names
    }
    PC_RANGE = [-40, -40, -8, 40, 40, 2]
    metadata = yaml.safe_load(sample["yaml"].read_text(encoding="utf-8"))
    grouped = {}
    for object_id, obj in metadata.get("vehicles", {}).items():
        super_name = INVERSE_CLASS_MAP.get(obj.get("obj_type"))
        if super_name is None:
            continue
        obj_copy = copy.deepcopy(obj)
        obj_copy["obj_type"] = np.asarray([super_name])
        grouped[object_id] = obj_copy
    projected = {}
    box_utils.project_world_objects(grouped, projected, metadata["lidar_pose"], PC_RANGE, "lwh")
    rows = []
    for box_and_class in projected.values():
        row = box_and_class[0]
        rows.append({"box": [float(x) for x in row[:7]], "class": str(row[7]).lower()})
    return rows


def infer_on_manifest(config: Path, checkpoint: Path, manifest: Path, output: Path,
                      device: int, gblobs: bool = False) -> None:
    if gblobs:
        sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "GBlobs" / "tools"),
                        str(PROJECT_ROOT / "third_party" / "GBlobs")]
    else:
        sys.path[:0] = [str(PROJECT_ROOT / "third_party" / "OpenPCDet" / "tools"),
                        str(PROJECT_ROOT / "third_party" / "OpenPCDet")]
    samples = []
    for m in load_manifest(manifest):
        bin_path = PROJECT_ROOT / m["path"]
        yaml_path = bin_path.with_suffix(".yaml")
        if not bin_path.exists() or not yaml_path.exists():
            raise SystemExit(f"missing files for {m['id']}: {bin_path} / {yaml_path}")
        samples.append({"id": m["id"], "bin": bin_path, "yaml": yaml_path})

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(config), cfg)
    apply_profile_local(cfg)
    dataset = LocalPublicICDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, samples, logger)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(str(checkpoint), logger=logger, to_cpu=True)
    torch.cuda.set_device(device)
    model.cuda().eval()

    predictions, ground_truth = {}, {}
    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            pred = model.forward(batch)[0][0]
            boxes = pred["pred_boxes"].cpu().numpy()
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            predictions[sample["id"]] = [
                {"box": [float(x) for x in box[:7]], "score": float(score),
                 "class": cfg.CLASS_NAMES[int(label) - 1]}
                for box, score, label in zip(boxes, scores, labels)
            ]
            ground_truth[sample["id"]] = ground_truth_for_sample_local(sample)
            if (index + 1) % 10 == 0:
                logger.info("infer %d/%d", index + 1, len(samples))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "samples": len(samples),
        "predictions": predictions,
        "ground_truth": ground_truth,
    }, indent=1), encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(predictions),
                      "boxes": sum(map(len, predictions.values()))}))


def iou3d(a, b):
    """Fast 3D IoU for axis-aligned boxes (rotation ignored) used for fusion."""
    ax1, ay1, az1 = a[:3] - a[3:6] / 2
    ax2, ay2, az2 = a[:3] + a[3:6] / 2
    bx1, by1, bz1 = b[:3] - b[3:6] / 2
    bx2, by2, bz2 = b[:3] + b[3:6] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = ix * iy * iz
    va = float(np.prod(a[3:6]))
    vb = float(np.prod(b[3:6]))
    return inter / max(1e-9, va + vb - inter)


def nms_fuse(frame_rows, nms_threshold, score_floor):
    out = []
    for cls in CLASSES:
        rows = sorted([r for r in frame_rows
                       if r["class"] == cls and r["score"] >= score_floor],
                      key=lambda r: r["score"], reverse=True)
        keep = []
        for r in rows:
            box = np.asarray(r["box"], dtype=float)
            if all(iou3d(box, np.asarray(k["box"], dtype=float)) < nms_threshold
                   for k in keep):
                keep.append(r)
        out.extend(keep)
    return out


def fuse_caches(caches, nms_threshold, score_floor, frame_ids):
    fused = {}
    for fid in frame_ids:
        rows = [row for cache in caches for row in cache["predictions"].get(fid, [])]
        fused[fid] = nms_fuse(rows, nms_threshold, score_floor)
    return fused


def run_fuse(args) -> None:
    caches = [json.loads(Path(p).read_text()) for p in args.cache]
    frame_ids = list(caches[0]["predictions"])
    gt = caches[0]["ground_truth"]
    report = []
    for nms in args.nms:
        for floor in args.score_floor:
            fused = fuse_caches(caches, nms, floor, frame_ids)
            d3 = evaluate_3d_detection(fused, gt, CLASSES, (0.5, 0.7), "3d")
            bev = evaluate_3d_detection(fused, gt, CLASSES, (0.5, 0.7), "bev")
            report.append({
                "nms": nms, "score_floor": floor,
                "n_pred": d3["n_pred"],
                "3d_05_R40": d3["mAP_0.5_R40"],
                "3d_07_R40": d3["mAP_0.7_R40"],
                "bev_05_R40": bev["mAP_0.5_R40"],
                "per_class": d3["per_class"],
            })
            print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    report.sort(key=lambda r: r["3d_05_R40"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("BEST:", json.dumps(report[0], ensure_ascii=False))


def run_export(args) -> None:
    """Apply the chosen fusion config to official test-frame predictions."""
    pred_sets = []
    for p in args.test_predictions:
        data = json.loads(Path(p).read_text())
        pred_sets.append(data)
    frame_ids = list(pred_sets[0])
    fused = fuse_caches(
        [{"predictions": p} for p in pred_sets],
        args.nms, args.score_floor, frame_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fused, indent=1), encoding="utf-8")
    print(json.dumps({"frames": len(fused),
                      "boxes": sum(map(len, fused.values())),
                      "output": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--gblobs", action="store_true", help="use GBlobs pcdet fork")

    p = sub.add_parser("fuse")
    p.add_argument("--cache", nargs="+", type=Path, required=True)
    p.add_argument("--nms", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3])
    p.add_argument("--score-floor", nargs="+", type=float, default=[0.03])
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("export")
    p.add_argument("--test-predictions", nargs="+", type=Path, required=True)
    p.add_argument("--nms", type=float, required=True)
    p.add_argument("--score-floor", type=float, default=0.03)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("infer-test")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--frame-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--gblobs", action="store_true", help="use GBlobs pcdet fork")

    args = parser.parse_args()
    if args.cmd == "infer":
        infer_on_manifest(args.config, args.checkpoint, args.manifest, args.output,
                          args.device, args.gblobs)
    elif args.cmd == "fuse":
        run_fuse(args)
    elif args.cmd == "export":
        run_export(args)
    elif args.cmd == "infer-test":
        infer_on_test_frames(args.config, args.checkpoint, args.frames, args.frame_ids,
                             args.output, args.device, args.gblobs)


if __name__ == "__main__":
    main()
