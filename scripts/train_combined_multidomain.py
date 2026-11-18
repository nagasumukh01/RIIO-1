"""
Train a Universal Multi-Domain Deepfake Classifier (EfficientNet-B0) on a balanced
combination of ALL 4 datasets in the project:
  1. FaceForensics++ (Deepfakes, Face2Face, NeuralTextures)
  2. Celeb-DF v2 (Celebrity Face Swaps)
  3. DFDC (Deepfake Detection Challenge)
  4. Diffusion Face Dataset (SDXL Synthetic Faces + CelebA-HQ Real)

Dataset Distribution:
  - Combined Train : 426 samples (213 Real, 213 Fake)
  - Combined Val   :  93 samples ( 46 Real,  47 Fake)
  - Combined Test  : 176 samples ( 88 Real,  88 Fake) [Held-out]

Saves checkpoint to: artifacts/models/combined_multidomain_b0/best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
logger = logging.getLogger("robustdeep.train_combined")


class CombinedMultiDomainDataset(Dataset):
    """
    Unified dataset supporting both video face crops (directory with frames)
    and single diffusion face images.
    """

    def __init__(self, samples: List[dict], transform=None, is_train: bool = True):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        target_path = Path(sample["path"])
        label = float(sample["label"])  # 0 = Real, 1 = Fake

        # If path is a directory (video frames)
        if target_path.is_dir():
            frames = sorted(target_path.glob("*.jpg")) or sorted(target_path.glob("*.png"))
            if not frames:
                raise FileNotFoundError(f"No frames found in {target_path}")
            # During training, pick a random frame; during eval pick the middle frame
            img_file = random.choice(frames) if self.is_train else frames[len(frames) // 2]
        else:
            img_file = target_path

        img = Image.open(img_file).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        return tensor, torch.tensor(label, dtype=torch.float32)


def load_all_dataset_samples(split_name: str) -> List[dict]:
    """Load and aggregate samples from all 4 datasets for a given split."""
    aggregated = []

    # 1-3. Video datasets: faceforensics, celebdf, dfdc
    for ds_name in ["faceforensics", "celebdf", "dfdc"]:
        manifest_path = PROJECT_ROOT / "data" / "processed" / ds_name / f"{split_name}_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            for s in samples:
                p = PROJECT_ROOT / s["path"]
                if p.exists():
                    aggregated.append({
                        "path": str(p),
                        "label": int(s["label"]),
                        "dataset": ds_name,
                        "split": split_name,
                    })

    # 4. Diffusion face dataset
    diff_manifest = PROJECT_ROOT / "data" / "diffusion_face" / "manifests" / f"{split_name}_manifest.json"
    if diff_manifest.exists():
        with open(diff_manifest, encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        for s in samples:
            p = Path(s["path"])
            if p.exists():
                aggregated.append({
                    "path": str(p),
                    "label": int(s["label"]),
                    "dataset": "diffusion_sdxl",
                    "split": split_name,
                })

    return aggregated


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def evaluate_dataset(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
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

    return mean_loss, acc, auc, y_true, y_score


def train_combined_model(epochs: int = 5, batch_size: int = 16, lr: float = 5e-5, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    train_samples = load_all_dataset_samples("train")
    val_samples = load_all_dataset_samples("val")
    test_samples = load_all_dataset_samples("test")

    train_tf, eval_tf = get_transforms()

    train_ds = CombinedMultiDomainDataset(train_samples, transform=train_tf, is_train=True)
    val_ds = CombinedMultiDomainDataset(val_samples, transform=eval_tf, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Print distribution
    print("\n" + "=" * 75)
    print("  TRAINING UNIVERSAL MULTI-DOMAIN DEEPFAKE CLASSIFIER")
    print("=" * 75)
    print("  COMBINED DATASET INVENTORY:")
    print(f"    - Train : {len(train_samples)} samples ({sum(1 for s in train_samples if s['label']==0)} Real, {sum(1 for s in train_samples if s['label']==1)} Fake)")
    print(f"    - Val   : {len(val_samples)} samples ({sum(1 for s in val_samples if s['label']==0)} Real, {sum(1 for s in val_samples if s['label']==1)} Fake)")
    print(f"    - Test  : {len(test_samples)} samples ({sum(1 for s in test_samples if s['label']==0)} Real, {sum(1 for s in test_samples if s['label']==1)} Fake) [HELD-OUT]")
    print("-" * 75)
    print(f"  Architecture : EfficientNet-B0 (Pretrained ImageNet)")
    print(f"  Epochs       : {epochs}  |  Batch size: {batch_size}  |  LR: {lr}")
    print("=" * 75)

    model = build_model(name="efficientnet_b0", pretrained=True, num_classes=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    # Scheduler: cosine annealing for stable convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    GRAD_CLIP = 1.0  # prevent gradient explosion (bug fix)

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "models" / "combined_multidomain_b0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient clipping
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()  # cosine LR decay

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, val_auc, _, _ = evaluate_dataset(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val AUC: %.4f",
            epoch, epochs, train_loss, val_loss, val_acc * 100.0, val_auc
        )

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
                "experiment_id": "combined_multidomain_b0",
                "dataset": "All 4 Combined (FF++, Celeb-DF, DFDC, Diffusion)",
                "architecture": "efficientnet_b0",
                "is_synthetic": False,
            }
            torch.save(ckpt, checkpoint_dir / "best.pt")

    print("-" * 75)
    print(f"  Training Finished! Best Epoch: {best_epoch} (Val AUC: {best_val_auc:.4f}, Val Acc: {best_val_acc*100:.1f}%)")
    print(f"  Saved checkpoint -> {checkpoint_dir / 'best.pt'}")

    # Comprehensive Per-Domain Evaluation on the 176 HELD-OUT TEST SAMPLES
    print("\n" + "=" * 75)
    print("  HELD-OUT TEST SET EVALUATION ACROSS ALL DOMAINS (176 SAMPLES)")
    print("=" * 75)

    best_ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    # 1. Overall test performance
    test_ds = CombinedMultiDomainDataset(test_samples, transform=eval_tf, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_loss, test_acc, test_auc, _, _ = evaluate_dataset(model, test_loader, criterion, device)

    # 2. Per-domain breakdown
    domain_reports = {}
    print(f"{'Domain / Dataset':<24} | {'Test Samples':<14} | {'Accuracy':<14} | {'ROC-AUC':<10}")
    print("-" * 75)

    for ds_name in ["faceforensics", "celebdf", "dfdc", "diffusion_sdxl"]:
        domain_samples = [s for s in test_samples if s["dataset"] == ds_name]
        if not domain_samples:
            continue
        dom_ds = CombinedMultiDomainDataset(domain_samples, transform=eval_tf, is_train=False)
        dom_loader = DataLoader(dom_ds, batch_size=batch_size, shuffle=False)
        _, d_acc, d_auc, _, _ = evaluate_dataset(model, dom_loader, criterion, device)
        domain_reports[ds_name] = {
            "samples": len(domain_samples),
            "accuracy": round(d_acc, 4),
            "auc": round(d_auc, 4),
        }
        print(f"{ds_name:<24} | {len(domain_samples):<14} | {d_acc*100:>11.1f}% | {d_auc:>9.4f}")

    print("-" * 75)
    print(f"{'OVERALL MULTI-DOMAIN':<24} | {len(test_samples):<14} | {test_acc*100:>11.1f}% | {test_auc:>9.4f}")
    print("=" * 75 + "\n")

    # Save artifact
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "model": "combined_multidomain_b0",
        "architecture": "efficientnet_b0",
        "datasets_combined": ["faceforensics", "celebdf", "dfdc", "diffusion_sdxl"],
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "overall_test": {
            "total_samples": len(test_samples),
            "test_accuracy": round(test_acc, 4),
            "test_auc": round(test_auc, 4),
            "test_loss": round(test_loss, 4),
        },
        "domain_breakdown": domain_reports,
        "checkpoint_path": str(checkpoint_dir / "best.pt"),
    }
    rpath = PROJECT_ROOT / "artifacts" / "experiment_logs" / f"combined_multidomain_train_{ts}.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train_combined_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
