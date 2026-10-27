#!/usr/bin/env python3
"""
RobustDeep training entry point.

CRITICAL: Loads REAL datasets only (faceforensics | celebdf | dfdc).
Raises FileNotFoundError with instructions if face crops are missing.
Synthetic data is NEVER used here.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset, collate_real
from src.models.factory import build_model, count_parameters
from src.training.trainer import Trainer
from src.evaluation.artifacts import export_evaluation_artifacts, predict_loader
from src.utils.config import get_data_config, get_training_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("train")


def _jpeg_reencode(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=40)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _build_augmentation(name: str):
    operations = []
    if name in ("jpeg", "combined"):
        operations.append(transforms.Lambda(_jpeg_reencode))
    if name in ("blur", "combined"):
        operations.append(transforms.Lambda(lambda image: image.filter(ImageFilter.GaussianBlur(radius=1.5))))
    if name in ("resize_reencode", "combined"):
        operations.append(transforms.Lambda(lambda image: image.resize((224, 224), Image.Resampling.BILINEAR)))
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(operations) if name != "none" else None


def parse_args():
    p = argparse.ArgumentParser(description="Train RobustDeep on a REAL dataset")
    p.add_argument("--dataset", type=str, required=True, choices=["faceforensics", "celebdf", "dfdc"],
                   help="Real dataset name. Synthetic is forbidden.")
    p.add_argument("--model", type=str, default="efficientnet_b0",
                   choices=["efficientnet_b0", "efficientnet_b4", "vit_small", "vit_base"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--max-videos", type=int, default=None, help="Limit videos for quick runs")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss", choices=["bce", "focal", "weighted_bce"], default=None)
    p.add_argument("--augmentation", choices=["none", "jpeg", "blur", "resize_reencode", "combined"], default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--experiment-id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data_cfg = get_data_config()
    train_cfg = get_training_config()
    if args.epochs is not None:
        train_cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["optimizer"]["lr"] = args.lr
    if args.loss is not None:
        train_cfg["loss"]["name"] = args.loss

    exp_id = args.experiment_id or f"{args.dataset}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Experiment ID: %s | dataset=%s | model=%s", exp_id, args.dataset, args.model)

    # REAL datasets only — will raise FileNotFoundError if faces missing
    train_transform = _build_augmentation(args.augmentation)
    train_ds = RealFaceVideoDataset(
        dataset=args.dataset,
        split="train",
        frames_per_video=args.frames,
        max_videos=args.max_videos,
        transform=train_transform,
    )
    val_ds = RealFaceVideoDataset(
        dataset=args.dataset,
        split="val",
        frames_per_video=args.frames,
        max_videos=args.max_videos,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_real,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_real,
        pin_memory=torch.cuda.is_available(),
    )
    test_ds = RealFaceVideoDataset(
        dataset=args.dataset,
        split="test",
        frames_per_video=args.frames,
        max_videos=args.max_videos,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_real,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        name=args.model,
        pretrained=True,
        aggregation=train_cfg["aggregation"]["method"],
        topk=train_cfg["aggregation"].get("topk", 5),
    )
    n_params = count_parameters(model)
    logger.info("Model parameters: %s", f"{n_params:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        experiment_id=exp_id,
        is_synthetic=False,
    )
    summary = trainer.fit()
    best_path = Path(trainer.ckpt_dir) / "best.pt"
    checkpoint = torch.load(best_path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_logits, val_labels, val_ids = predict_loader(model, val_loader, trainer.device)
    test_logits, test_labels, test_ids = predict_loader(model, test_loader, trainer.device)
    val_artifacts = export_evaluation_artifacts(
        val_logits, val_labels, val_ids, "val", args.dataset, trainer.ckpt_dir,
        {"experiment_id": exp_id, "model": args.model},
    )
    test_artifacts = export_evaluation_artifacts(
        test_logits, test_labels, test_ids, "test", args.dataset, trainer.ckpt_dir,
        {"experiment_id": exp_id, "model": args.model},
    )
    summary.update({
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(best_path),
        "val_metrics": val_artifacts["metrics"],
        "test_metrics": test_artifacts["metrics"],
        "artifact_dir": str(trainer.ckpt_dir),
    })
    with (Path(trainer.ckpt_dir) / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True, default=str)
    experiment_log = Path(trainer.ckpt_dir).parents[1] / "experiment_logs" / f"{exp_id}.json"
    experiment_log.parent.mkdir(parents=True, exist_ok=True)
    with experiment_log.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True, default=str)
    logger.info("Done. Best val AUROC: %.4f | label=%s", summary["best_val_auroc"], summary["label"])
    return summary


if __name__ == "__main__":
    main()
