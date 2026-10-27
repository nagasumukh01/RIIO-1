#!/usr/bin/env python3
"""Generate labeled attribution images and a lightweight faithfulness report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.real_dataset import RealFaceVideoDataset, collate_real
from src.explainability.gradcam import GradCAM
from src.models.factory import build_model


def _gradient_map(model, frame):
    frame = frame.unsqueeze(0).requires_grad_(True)
    model.zero_grad()
    model.backbone(frame).sum().backward()
    saliency = frame.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="faceforensics", choices=["faceforensics", "celebdf", "dfdc"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--max-videos", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    output = Path(args.out) if args.out else ROOT / "artifacts" / "figures" / "explainability" / Path(args.checkpoint).parent.name
    output.mkdir(parents=True, exist_ok=True)
    dataset = RealFaceVideoDataset(args.dataset, args.split, max_videos=args.max_videos)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_real)
    model = build_model(args.model, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cam = None
    if getattr(model, "backbone_type", "cnn") == "cnn":
        layers = [module for module in model.backbone.modules() if isinstance(module, torch.nn.Conv2d)]
        if layers:
            cam = GradCAM(model, layers[-1])
    records = []
    for batch in loader:
        video = batch["video"]
        with torch.no_grad():
            score = float(torch.sigmoid(model(video))[0])
        label = int(batch["label"][0])
        prediction = int(score >= 0.5)
        category = "correct" if prediction == label else ("false_positive" if prediction else "false_negative")
        uncertainty = abs(score - 0.5)
        attribution = cam(video[:, 0]) if cam is not None else _gradient_map(model, video[:, 0])
        image = video[0, 0].permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(image)
        axes[1].imshow(attribution, cmap="magma", alpha=0.55)
        axes[1].set_title("Model attribution, not forensic proof")
        axes[1].axis("off")
        figure.suptitle(f"{category} | score={score:.3f} | {batch['video_id'][0]}")
        figure.tight_layout()
        path = output / f"{batch['video_id'][0]}_{category}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        masked = video.clone()
        mask = torch.from_numpy(attribution >= np.quantile(attribution, 0.9))
        masked[0, 0, :, mask] = 0
        with torch.no_grad():
            masked_score = float(torch.sigmoid(model(masked))[0])
        records.append({"video_id": batch["video_id"][0], "label": label, "score": score, "category": category, "uncertainty": uncertainty, "faithfulness_confidence_drop": score - masked_score, "figure": str(path), "label_text": "model attribution, not forensic proof"})
    if cam is not None:
        cam.remove_hooks()
    (output / "report.json").write_text(json.dumps({"checkpoint": args.checkpoint, "dataset": args.dataset, "split": args.split, "records": records, "note": "Visualizations are model attribution, not forensic proof."}, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Explainability report written to {output}")


if __name__ == "__main__":
    main()
