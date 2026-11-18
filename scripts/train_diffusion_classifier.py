"""
Train an EfficientNet-B0 classifier on the balanced Diffusion Face Dataset (Contribution 1).

Dataset:
  - Train: 70 samples (35 Real [CelebA-HQ] + 35 Fake [SDXL])
  - Val:   14 samples (7 Real + 7 Fake)
  - Test:  16 samples (8 Real + 8 Fake)

Saves checkpoint to: artifacts/models/diffusion_b0_balanced70/best.pt
Registers model as 6th ensemble member in configs/ensemble.yaml.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import build_model
from src.utils.logging_utils import setup_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("robustdeep.train_diffusion")


class DiffusionFaceDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None):
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data.get("samples", [])
        self.transform = transform
        logger.info(
            "Loaded %d samples from %s (%d Real, %d Fake)",
            len(self.samples),
            manifest_path.name,
            sum(1 for s in self.samples if s["label"] == 0),
            sum(1 for s in self.samples if s["label"] == 1),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = Path(sample["path"])
        label = float(sample["label"])  # 0.0 = Real, 1.0 = Fake

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        return tensor, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1), targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits.view(-1)).cpu().numpy().tolist()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().tolist())

    n = max(len(loader.dataset), 1)
    mean_loss = total_loss / n
    y_true = np.array(all_targets)
    y_score = np.array(all_preds)
    y_pred = (y_score >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.5

    return mean_loss, acc, auc


def train_diffusion_model(epochs: int = 5, batch_size: int = 8, lr: float = 1e-4, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    manifest_dir = PROJECT_ROOT / "data" / "diffusion_face" / "manifests"
    train_manifest = manifest_dir / "train_manifest.json"
    val_manifest = manifest_dir / "val_manifest.json"
    test_manifest = manifest_dir / "test_manifest.json"

    train_tf, eval_tf = get_transforms()

    train_ds = DiffusionFaceDataset(train_manifest, transform=train_tf)
    val_ds = DiffusionFaceDataset(val_manifest, transform=eval_tf)
    test_ds = DiffusionFaceDataset(test_manifest, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info("Initializing EfficientNet-B0 backbone (pretrained=True)...")
    model = build_model(name="efficientnet_b0", pretrained=True, num_classes=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "models" / "diffusion_b0_balanced70"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("  TRAINING EFFICIENTNET-B0 ON BALANCED DIFFUSION FACES")
    print("=" * 70)
    print(f"  Architecture : EfficientNet-B0 (timm)")
    print(f"  Training set : 70 samples (35 Real, 35 Fake)")
    print(f"  Val set      : 14 samples (7 Real, 7 Fake)")
    print(f"  Test set     : 16 samples (8 Real, 8 Fake - untouched)")
    print(f"  Epochs       : {epochs}  |  Batch Size: {batch_size}  |  LR: {lr}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val AUC: %.4f",
            epoch, epochs, train_loss, val_loss, val_acc * 100.0, val_auc
        )

        # Save best model based on val AUC and Acc
        if val_auc > best_val_auc or (val_auc == best_val_auc and val_acc >= best_val_acc):
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": {
                    "val_auc": float(val_auc),
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                },
                "experiment_id": "diffusion_b0_balanced70",
                "dataset": "Diffusion Face (SDXL + CelebA-HQ)",
                "architecture": "efficientnet_b0",
                "is_synthetic": False,
            }
            torch.save(ckpt, checkpoint_dir / "best.pt")

    print("-" * 70)
    print(f"  Training Complete! Best Epoch: {best_epoch} (Val AUC: {best_val_auc:.4f}, Val Acc: {best_val_acc*100:.1f}%)")
    print(f"  Saved checkpoint -> {checkpoint_dir / 'best.pt'}")

    # Evaluate best checkpoint on held-out test split
    logger.info("Evaluating best model on held-out TEST split (16 samples)...")
    best_ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("  HELD-OUT TEST SET EVALUATION RESULTS (16 SAMPLES)")
    print("=" * 70)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc*100:.1f}%")
    print(f"  Test ROC-AUC  : {test_auc:.4f}")
    print("=" * 70 + "\n")

    # Save training report artifact
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "model": "diffusion_b0",
        "backbone": "efficientnet_b0",
        "dataset": "diffusion_face_balanced",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_auc": round(test_auc, 4),
        "test_loss": round(test_loss, 4),
        "checkpoint_path": str(checkpoint_dir / "best.pt"),
    }
    report_path = PROJECT_ROOT / "artifacts" / "experiment_logs" / f"diffusion_b0_train_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_diffusion_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
