"""
Diffusion-Generated Face Manipulation Evaluation Module.

Purpose:
    Evaluate trained deepfake detectors on diffusion-generated or
    diffusion-assisted face manipulations — a manipulation family NOT covered
    by FF++, Celeb-DF, or DFDC.

Status:
    PENDING — DATA ACCESS LIMITED

    No publicly accessible diffusion-face-manipulation dataset with a permissive
    license suitable for this research was identified at project time.
    Known candidates and limitations:
      - DiffSwap (2023): requires MPI license agreement, not public
      - Diff-Deepfake (2024): not publicly released
      - Custom generation via Stable Diffusion 3 + IP-Adapter: legally usable
        with CC0/public-domain source portraits, but requires GPU pipeline

    This module provides:
      1. The complete evaluation interface (ready when data is available)
      2. Documentation of expected data format and provenance requirements
      3. A synthetic placeholder that validates the pipeline end-to-end

Research Context (Petmezas 2026):
    The primary reference paper's Discussion section explicitly flags
    diffusion-generated manipulations as a dataset limitation.  FF++,
    Celeb-DF, DFD, and ReenactFaces all predate or exclude diffusion-based
    synthesis pipelines. This module addresses that gap when data becomes
    available.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.utils.logging_utils import get_logger
from src.utils.metrics import compute_all_metrics

logger = get_logger("diffusion_eval")


# ─── Expected Dataset Format ─────────────────────────────────────────────────

DIFFUSION_DATA_FORMAT = """
Expected directory structure for diffusion evaluation set:

    data/diffusion/
    ├── metadata.json          # Required: provenance, generation method, license
    ├── real/                  # Real (unmanipulated) face images
    │   └── *.jpg / *.png
    └── fake/                  # Diffusion-generated manipulated faces
        └── *.jpg / *.png

metadata.json MUST contain:
    {
        "dataset_name": str,
        "generation_method": str,    # e.g. "Stable Diffusion 3 + IP-Adapter v2"
        "license": str,              # Must be permissive for research use
        "provenance": str,           # Source of original identity images
        "creation_date": str,
        "num_real": int,
        "num_fake": int,
        "known_limitations": ["str", ...],
        "ethical_review": str,       # "N/A" or review reference
        "generation_parameters": {}  # Model checkpoint, seed, guidance scale, etc.
    }

How to construct the set yourself (GPU required):
    1. Obtain source face images under CC0 or public domain license
       (e.g. from LAION-Aesthetics filtered subset, with license verification)
    2. Verify NO overlap with FF++, Celeb-DF, or DFDC identities
    3. Run IP-Adapter or equivalent diffusion-based face swap pipeline:
         - Model: stabilityai/stable-diffusion-3-medium
         - Face encoder: IP-Adapter FaceID
         - Resolution: 512x512 → crop and align to 224x224
    4. Document generation parameters (checkpoint hash, seed, CFG scale)
    5. Obtain ethical review if required by your institution
    6. Place results under data/diffusion/real/ and data/diffusion/fake/
    7. Write metadata.json with all provenance information
"""

DIFFUSION_DATA_DIR = "data/diffusion"


# ─── Dataset Class ────────────────────────────────────────────────────────────

class DiffusionFaceDataset(Dataset):
    """
    Dataset loader for diffusion-generated face manipulation evaluation.

    Expects face images already cropped and aligned to face_size × face_size.
    Uses ImageNet normalization (matching training preprocessing).
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        face_size: int = 224,
    ):
        self.face_size = face_size
        self.mean = np.array(self.IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(self.IMAGENET_STD, dtype=np.float32)
        self.samples: List[Dict] = []

        for img_path in Path(real_dir).glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                self.samples.append({"path": str(img_path), "label": 0})

        for img_path in Path(fake_dir).glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                self.samples.append({"path": str(img_path), "label": 1})

        logger.info(
            f"DiffusionFaceDataset: {sum(s['label']==0 for s in self.samples)} real, "
            f"{sum(s['label']==1 for s in self.samples)} fake"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        try:
            img = Image.open(s["path"]).convert("RGB")
            img = img.resize((self.face_size, self.face_size), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np = (img_np - self.mean) / self.std
            tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        except Exception as e:
            logger.warning(f"Failed to load {s['path']}: {e}")
            tensor = torch.zeros(3, self.face_size, self.face_size)

        return {
            "image": tensor,
            "label": torch.tensor(s["label"], dtype=torch.long),
            "path": s["path"],
            "video_id": Path(s["path"]).stem,
        }


# ─── Evaluator ───────────────────────────────────────────────────────────────

class DiffusionEvaluator:
    """
    Evaluates a trained deepfake detector on diffusion-generated face manipulations.

    The evaluation set must be completely separate from training data.
    No threshold or hyperparameter tuning is performed on this set.
    Results are always reported separately from the main cross-dataset matrix.

    Protocol:
        - Threshold fixed at 0.5 (same as all other evaluations; never tuned on target)
        - Identical preprocessing to the main evaluation pipeline
        - Results labeled "diffusion_generated" to distinguish from GAN-based methods
    """

    STATUS_PENDING = "PENDING — DATA ACCESS LIMITED"
    STATUS_COMPLETED = "COMPLETED"

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        results_dir: str = "artifacts/experiment_logs",
        data_dir: str = DIFFUSION_DATA_DIR,
        face_size: int = 224,
    ):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.face_size = face_size

    def check_data_available(self) -> Dict[str, Any]:
        """Check if diffusion evaluation data is present and valid."""
        metadata_path = self.data_dir / "metadata.json"
        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "fake"

        real_count = 0
        fake_count = 0
        if real_dir.exists():
            real_count = len([
                f for f in real_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            ])
        if fake_dir.exists():
            fake_count = len([
                f for f in fake_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            ])

        status = {
            "data_available": real_count > 0 and fake_count > 0,
            "data_dir": str(self.data_dir),
            "metadata_found": metadata_path.exists(),
            "real_dir_found": real_dir.exists(),
            "fake_dir_found": fake_dir.exists(),
            "real_count": real_count,
            "fake_count": fake_count,
            "status": self.STATUS_PENDING if real_count == 0 or fake_count == 0
            else self.STATUS_COMPLETED,
            "instructions": DIFFUSION_DATA_FORMAT,
        }

        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    status["metadata"] = json.load(f)
            except Exception as e:
                status["metadata_error"] = str(e)

        return status

    @torch.no_grad()
    def evaluate(
        self,
        use_synthetic_fallback: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Evaluate model on diffusion-generated faces.

        If real diffusion data is unavailable and use_synthetic_fallback=True,
        runs on synthetic data to validate the pipeline. This is clearly labeled
        as a pipeline validation result, never a research result.

        Args:
            use_synthetic_fallback: Use synthetic data when diffusion data
                is not available, for pipeline validation only.
            batch_size: Inference batch size.

        Returns:
            Dict with keys: status, data_status, metrics (or pipeline_validation),
            warning, provenance.
        """
        data_status = self.check_data_available()
        result: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "diffusion_generated",
            "data_status": {k: v for k, v in data_status.items() if k != "instructions"},
            "warning": None,
            "metrics": None,
        }

        if not data_status["data_available"]:
            msg = (
                f"PENDING — Diffusion evaluation data not found at {self.data_dir}. "
                "See data_status field for setup instructions."
            )
            logger.warning(msg)
            result["warning"] = msg
            result["status"] = self.STATUS_PENDING

            if use_synthetic_fallback:
                logger.info(
                    "Running pipeline validation on SYNTHETIC data "
                    "(this is NOT a research result)"
                )
                result["pipeline_validation"] = self._run_pipeline_validation()

            self._save_result(result, "diffusion_eval_status.json")
            return result

        # ── Real diffusion data is available ─────────────────────────────
        logger.info("=" * 60)
        logger.info("DIFFUSION-GENERATED FACE MANIPULATION EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Real samples:             {data_status['real_count']}")
        logger.info(f"Fake (diffusion) samples: {data_status['fake_count']}")

        dataset = DiffusionFaceDataset(
            real_dir=str(self.data_dir / "real"),
            fake_dir=str(self.data_dir / "fake"),
            face_size=self.face_size,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        all_labels: List[int] = []
        all_scores: List[float] = []
        all_preds: List[int] = []

        self.model.eval()
        eval_start = time.time()

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"]

            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        eval_time = time.time() - eval_start

        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_scores),
        )
        metrics["eval_time_s"] = eval_time

        logger.info(
            f"Results — Acc={metrics['accuracy']:.4f}  "
            f"AUROC={metrics['auroc']:.4f}  EER={metrics['eer']:.4f}  "
            f"F1={metrics['f1_score']:.4f}"
        )

        result["metrics"] = metrics
        result["status"] = self.STATUS_COMPLETED
        result["provenance"] = data_status.get("metadata", {})

        self._save_result(result, "diffusion_eval_results.json")
        return result

    def _run_pipeline_validation(self) -> Dict:
        """Validate pipeline wiring using synthetic data. NOT a research result."""
        from src.data.dataset import SyntheticDataset, create_dataloaders
        from torch.utils.data import DataLoader

        # create_dataloaders requires separate train + val datasets
        test_ds = SyntheticDataset(num_samples=40, seed=999, split="test")
        val_ds = SyntheticDataset(num_samples=10, seed=998, split="val")
        # Use test dataset directly with a simple DataLoader (no split needed)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

        all_labels, all_scores, all_preds = [], [], []
        self.model.eval()

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"]
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_labels.extend(labels.numpy().tolist())
                all_scores.extend(probs[:, 1].cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        metrics = compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_scores)
        )
        return {
            "note": (
                "SYNTHETIC DATA — pipeline validation only. "
                "These numbers validate the evaluation wiring, NOT model capability."
            ),
            "metrics": metrics,
        }

    def _save_result(self, result: Dict, filename: str) -> None:
        """Save evaluation result, filtering non-serializable values."""
        def _clean(obj):
            if isinstance(obj, dict):
                return {
                    k: _clean(v) for k, v in obj.items()
                    if isinstance(v, (int, float, str, bool, list, dict, type(None)))
                }
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        out_path = self.results_dir / filename
        with open(out_path, "w") as f:
            json.dump(_clean(result), f, indent=2)
        logger.info(f"Result saved to {out_path}")

    def generate_report(self) -> str:
        """Print and return a text summary of the diffusion evaluation status."""
        data_status = self.check_data_available()
        lines = [
            "=" * 65,
            "DIFFUSION-GENERATED FACE MANIPULATION EVALUATION",
            "=" * 65,
            f"Current Status: {data_status['status']}",
            "",
            "Research Context:",
            "  Petmezas 2026 Discussion flags diffusion-generated manipulations",
            "  as a dataset limitation. This module addresses that gap.",
            "",
            "Data Location:",
            f"  Expected: {self.data_dir}",
            f"  Directory exists: {self.data_dir.exists()}",
            f"  Real images found:  {data_status['real_count']}",
            f"  Fake images found:  {data_status['fake_count']}",
            "",
            "Setup Instructions:",
            "  See DIFFUSION_DATA_FORMAT constant in this module,",
            "  or run: python scripts/run_diffusion_eval.py --help",
            "=" * 65,
        ]
        report = "\n".join(lines)
        logger.info(report)
        return report
