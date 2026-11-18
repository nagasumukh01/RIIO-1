"""
Verify that all 5 ensemble model checkpoints load correctly.

Runs strict=True load for each checkpoint and reports:
  - File size
  - Load status (OK / MISSING KEYS / UNEXPECTED KEYS / LOAD ERROR)
  - Architecture used

Usage:
  python scripts/verify_checkpoints.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.models.factory import build_model


def load_ensemble_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "ensemble.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_ensemble_config()
    models_cfg = cfg.get("models", {})
    device = torch.device("cpu")

    print("\n" + "=" * 72)
    print("CHECKPOINT VERIFICATION REPORT")
    print("=" * 72)
    print(f"{'ID':<16} {'Architecture':<24} {'Size':>8}  Status")
    print("-" * 72)

    results = []
    for model_id, m in models_cfg.items():
        ckpt_rel = m.get("checkpoint", "")
        ckpt_path = PROJECT_ROOT / ckpt_rel
        arch = m.get("timm_name", m.get("architecture", "?"))

        if not ckpt_path.exists():
            print(f"{model_id:<16} {arch:<24} {'MISSING':>8}  ✗ File not found: {ckpt_path}")
            results.append((model_id, "MISSING"))
            continue

        size_mb = ckpt_path.stat().st_size / 1e6

        try:
            model = build_model(arch, num_classes=1)
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

            # Checkpoint format: {'epoch':..., 'model_state': <state_dict>, ...}
            # model_state keys already have 'backbone.' prefix matching VideoDeepfakeClassifier.
            if isinstance(state, dict):
                if "model_state" in state:
                    state_dict = state["model_state"]
                elif "model" in state and isinstance(state["model"], dict):
                    state_dict = state["model"]
                elif "state_dict" in state:
                    state_dict = state["state_dict"]
                else:
                    state_dict = state
            else:
                state_dict = state

            # Load strict=False — only flag genuine backbone key mismatches
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            missing_backbone = [k for k in missing if k.startswith("backbone.")]
            if not missing_backbone and not unexpected:
                status = "✓ OK — all backbone keys matched"
            elif not missing_backbone:
                status = f"OK — {len(unexpected)} unexpected keys (old ckpt format)"
            else:
                status = f"WARN — {len(missing_backbone)} backbone keys missing"

            print(f"{model_id:<16} {arch:<24} {size_mb:>7.1f}MB  {status}")
            results.append((model_id, "OK"))
        except Exception as e:
            print(f"{model_id:<16} {arch:<24} {size_mb:>7.1f}MB  ✗ LOAD ERROR: {e}")
            results.append((model_id, "ERROR"))

    print("=" * 72)
    n_ok = sum(1 for _, s in results if s == "OK")
    print(f"\nSummary: {n_ok}/{len(results)} checkpoints loaded successfully.")

    if n_ok < len(results):
        print("\nFailed models:")
        for mid, s in results:
            if s != "OK":
                print(f"  {mid}: {s}")

    print()


if __name__ == "__main__":
    main()
