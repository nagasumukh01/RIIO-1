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
from src.explainability.attention_rollout import attention_rollout
from src.explainability.gradcam import GradCAM
from src.models.factory import build_model


def _gradient_map(model, frame):
    frame = frame.unsqueeze(0).requires_grad_(True)
    model.zero_grad()
    model.backbone(frame).sum().backward()
    saliency = frame.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


def _install_attention_capture(model):
    """Capture per-block self-attention weights from standard timm ViT blocks."""
    captures = []
    originals = []
    backbone = model.backbone
    for block in getattr(backbone, "blocks", []):
        attention = getattr(block, "attn", None)
        if attention is None or not all(hasattr(attention, name) for name in ("qkv", "proj")):
            continue
        originals.append((attention, attention.forward))

        def forward(x, attn_mask=None, is_causal=False, attention=attention, **kwargs):
            batch, tokens, channels = x.shape
            heads = attention.num_heads
            head_dim = channels // heads
            qkv = attention.qkv(x).reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            weights = (query @ key.transpose(-2, -1)) * attention.scale
            weights = weights.softmax(dim=-1)
            captures.append(weights.detach())
            weights = attention.attn_drop(weights)
            output = (weights @ value).transpose(1, 2).reshape(batch, tokens, channels)
            return attention.proj_drop(attention.proj(output))

        attention.forward = forward
    return captures, originals


def _remove_attention_capture(originals):
    for attention, original in originals:
        attention.forward = original


def _attention_map(model, frame, captures):
    captures.clear()
    with torch.no_grad():
        model.backbone(frame)
    if not captures:
        return _gradient_map(model, frame), "gradient attribution"
    tokens = attention_rollout(captures)
    side = int(np.sqrt(tokens.shape[0]))
    grid = torch.from_numpy(tokens.reshape(side, side)).float()[None, None]
    grid = torch.nn.functional.interpolate(grid, size=frame.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    return grid, "attention rollout"


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
    attention_captures, attention_originals = [], []
    if getattr(model, "backbone_type", "cnn") == "cnn":
        layers = [module for module in model.backbone.modules() if isinstance(module, torch.nn.Conv2d)]
        if layers:
            cam = GradCAM(model, layers[-1])
    else:
        attention_captures, attention_originals = _install_attention_capture(model)
    records = []
    for batch in loader:
        video = batch["video"]
        with torch.no_grad():
            score = float(torch.sigmoid(model(video))[0])
        label = int(batch["label"][0])
        prediction = int(score >= 0.5)
        category = "correct" if prediction == label else ("false_positive" if prediction else "false_negative")
        uncertainty = abs(score - 0.5)
        if cam is not None:
            attribution = cam(video[:, 0])
            attribution_method = "Grad-CAM"
        else:
            attribution, attribution_method = _attention_map(model, video[:, 0], attention_captures)
        image = video[0, 0].permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(image)
        axes[1].imshow(attribution, cmap="magma", alpha=0.55)
        axes[1].set_title(f"{attribution_method}; not forensic proof")
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
        records.append({"video_id": batch["video_id"][0], "label": label, "score": score, "category": category, "uncertainty": uncertainty, "faithfulness_confidence_drop": score - masked_score, "figure": str(path), "method": attribution_method, "label_text": "model attribution, not forensic proof"})
    if cam is not None:
        cam.remove_hooks()
    _remove_attention_capture(attention_originals)
    (output / "report.json").write_text(json.dumps({"checkpoint": args.checkpoint, "dataset": args.dataset, "split": args.split, "records": records, "note": "Visualizations are model attribution, not forensic proof."}, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Explainability report written to {output}")


if __name__ == "__main__":
    main()
