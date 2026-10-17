"""
Training engine for deepfake detection models.
Supports configurable training loops, early stopping, checkpointing.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from src.utils.config import TrainingConfig
from src.utils.metrics import compute_all_metrics
from src.utils.logging_utils import get_logger
from src.training.losses import create_loss_function

logger = get_logger("trainer")


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(self, patience: int = 7, mode: str = 'max', min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Model trainer with full experiment tracking.
    Records all training metrics, configurations, and predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cpu",
        experiment_id: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Created after we know epochs

        # Loss function
        self.criterion = create_loss_function(
            loss_type=config.loss_function,
            focal_gamma=config.focal_loss_gamma,
            focal_alpha=config.focal_loss_alpha,
            device=device,
        )

        # Early stopping
        self.early_stopping = None
        if config.early_stopping_enabled:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                mode=config.early_stopping_mode,
                min_delta=config.early_stopping_min_delta,
            )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auroc': [],
            'val_f1': [],
            'learning_rates': [],
            'epoch_times': [],
        }

        self.best_metric = 0.0
        self.best_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        params = self.model.parameters()
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=tuple(self.config.betas),
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            return torch.optim.AdamW(params, lr=self.config.learning_rate)

    def _create_scheduler(self, num_epochs: int):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', patience=3, factor=0.5,
            )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_scores = []
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Track metrics
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs[:, 1].detach().cpu().numpy())
            num_batches += 1

            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / max(num_batches, 1)
        metrics = compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_scores)
        )
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_scores = []
        num_batches = 0

        for batch in val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        metrics = compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_scores)
        )
        metrics['loss'] = avg_loss

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict:
        """Full training loop with validation, early stopping, and checkpointing."""
        if num_epochs is None:
            num_epochs = self.config.epochs

        self._create_scheduler(num_epochs)

        checkpoint_dir = Path(self.config.checkpoint_dir) / self.experiment_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training: {num_epochs} epochs, device={self.device}")
        logger.info(f"Experiment ID: {self.experiment_id}")

        training_start = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            logger.info(f"Epoch [{epoch}/{num_epochs}]")
            train_metrics = self.train_epoch(train_loader)
            logger.info(
                f"  Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}, "
                f"AUROC: {train_metrics['auroc']:.4f}"
            )

            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(
                f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"AUROC: {val_metrics['auroc']:.4f}, "
                f"F1: {val_metrics['f1_score']:.4f}"
            )

            epoch_time = time.time() - epoch_start

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(epoch_time)

            # Check for best model
            monitor_value = val_metrics.get(
                self.config.early_stopping_monitor.replace('val_', ''), 0
            )
            if monitor_value > self.best_metric:
                self.best_metric = monitor_value
                self.best_epoch = epoch

                # Save best checkpoint
                from src.models.factory import save_checkpoint
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    config={'training': vars(self.config), 'experiment_id': self.experiment_id},
                    path=str(checkpoint_dir / "best_model.pt"),
                )
                logger.info(f"  [*] New best model! {self.config.early_stopping_monitor}: {monitor_value:.4f}")

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(monitor_value):
                    logger.info(f"  Early stopping triggered at epoch {epoch}")
                    break

        total_time = time.time() - training_start
        logger.info(f"Training complete! Best epoch: {self.best_epoch}, "
                    f"Best {self.config.early_stopping_monitor}: {self.best_metric:.4f}, "
                    f"Total time: {total_time:.1f}s")

        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return {
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'total_time_s': total_time,
            'history': self.history,
            'checkpoint_dir': str(checkpoint_dir),
        }
