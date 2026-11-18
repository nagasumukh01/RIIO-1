"""Training loop with mixed precision, early stopping, checkpointing."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.training.losses import get_loss
from src.utils.config import PROJECT_ROOT, ensure_dir, get_training_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("robustdeep.trainer")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        experiment_id: str = "default",
        is_synthetic: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cfg = config or get_training_config()
        self.experiment_id = experiment_id
        self.is_synthetic = is_synthetic

        opt_cfg = self.cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        self.loss_fn = get_loss(self.cfg["loss"]["name"], focal_gamma=self.cfg["loss"].get("focal_gamma", 2.0))
        self.scaler = GradScaler(enabled=self.cfg["training"].get("mixed_precision", True) and self.device.type == "cuda")
        self.epochs = self.cfg["training"]["epochs"]
        self.grad_clip = self.cfg["training"].get("gradient_clip", 1.0)
        self.patience = self.cfg["training"].get("early_stopping_patience", 7)
        checkpoint_root = self.cfg["checkpoint"]["dir"] if is_synthetic else "artifacts/models"
        self.ckpt_dir = ensure_dir(PROJECT_ROOT / checkpoint_root / experiment_id)
        self.best_metric = -np.inf
        self.epochs_no_improve = 0
        self.history: list[dict] = []

    def _step(self, batch: dict, train: bool = True) -> tuple[float, torch.Tensor, torch.Tensor]:
        video = batch["video"].to(self.device)
        label = batch["label"].to(self.device)
        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(video)
                loss = self.loss_fn(logits, label)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.model.eval()
            with torch.no_grad(), autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(video)
                loss = self.loss_fn(logits, label)
        return loss.item(), logits.detach().cpu(), label.detach().cpu()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        losses, all_logits, all_labels = [], [], []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]", leave=False)
        for batch in pbar:
            loss, logits, labels = self._step(batch, train=True)
            losses.append(loss)
            all_logits.append(logits)
            all_labels.append(labels)
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
        metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
        metrics["loss"] = float(np.mean(losses))
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        losses, all_logits, all_labels = [], [], []
        for batch in tqdm(self.val_loader, desc="[val]", leave=False):
            loss, logits, labels = self._step(batch, train=False)
            losses.append(loss)
            all_logits.append(logits)
            all_labels.append(labels)
        metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
        metrics["loss"] = float(np.mean(losses))
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> Path:
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "experiment_id": self.experiment_id,
            "is_synthetic": self.is_synthetic,
        }
        torch.save(payload, path)
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(payload, best_path)
            logger.info("New best checkpoint saved → %s (auroc=%.4f)", best_path, metrics.get("auroc", -1))
        return path

    def fit(self) -> Dict[str, Any]:
        logger.info("Starting training on device=%s | synthetic=%s | experiment=%s", self.device, self.is_synthetic, self.experiment_id)
        start = time.time()
        for epoch in range(self.epochs):
            train_m = self.train_epoch(epoch)
            val_m = self.validate()
            record = {"epoch": epoch, "train": train_m, "val": val_m}
            self.history.append(record)
            monitor = val_m.get("auroc", train_m.get("auroc", -1))
            is_best = monitor > self.best_metric
            if is_best:
                self.best_metric = monitor
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            self.save_checkpoint(epoch, val_m or train_m, is_best=is_best)
            logger.info(
                "Epoch %d | train_loss=%.4f auroc=%.4f | val_auroc=%.4f | best=%.4f",
                epoch + 1,
                train_m["loss"],
                train_m.get("auroc", -1),
                val_m.get("auroc", -1) if val_m else -1,
                self.best_metric,
            )
            if self.epochs_no_improve >= self.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        elapsed = time.time() - start
        summary = {
            "experiment_id": self.experiment_id,
            "best_val_auroc": float(self.best_metric),
            "epochs_ran": len(self.history),
            "elapsed_seconds": elapsed,
            "is_synthetic": self.is_synthetic,
            "history": self.history,
            "label": "SYNTHETIC – NOT A RESEARCH RESULT" if self.is_synthetic else "REAL DATA",
            "config": self.cfg,
        }
        out = self.ckpt_dir / "training_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Training finished. Summary → %s", out)
        return summary
