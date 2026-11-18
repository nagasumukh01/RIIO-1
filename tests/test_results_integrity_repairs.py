from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from scripts.cross_dataset_matrix import predict_on_samples
from scripts.generate_results_export import (
    collect_diffusion_evaluation,
    collect_model_evaluations,
)


def test_cross_dataset_prediction_keeps_labels_aligned_after_skip(tmp_path: Path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (2, 2)).save(image_path)

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, _image):
            return torch.tensor([2.0])

    labels, scores = predict_on_samples(
        FixedModel(),
        [(str(tmp_path / "missing.png"), 0), (str(image_path), 1)],
        transform=lambda _image: torch.zeros((3, 2, 2)),
    )

    assert labels.tolist() == [1]
    assert scores.shape == (1,)


def test_diffusion_export_recomputes_correctness_against_ground_truth(tmp_path: Path):
    artifact_path = tmp_path / "diffusion_test_eval.json"
    artifact_path.write_text(json.dumps({
        "ensemble": {"accuracy": 1.0, "correct": 2, "total": 2},
        "per_sample_results": [
            {
                "ground_truth": "REAL",
                "ensemble_verdict": "REAL",
                "models": {"model_a": {"pred": "REAL"}},
            },
            {
                "ground_truth": "FAKE",
                "ensemble_verdict": "REAL",
                "models": {"model_a": {"pred": "REAL"}},
            },
        ],
    }), encoding="utf-8")

    result, error = collect_diffusion_evaluation(artifact_path)

    assert error is None
    assert result is not None
    assert result["ensemble"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert result["models"][0]["metrics"] == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_diffusion_export_marks_unsupported_verdicts_pending_without_scores(tmp_path: Path):
    artifact_path = tmp_path / "diffusion_test_eval.json"
    artifact_path.write_text(json.dumps({
        "per_sample_results": [{
            "ground_truth": "FAKE",
            "ensemble_verdict": "UNCERTAIN",
            "models": {},
        }],
    }), encoding="utf-8")

    result, error = collect_diffusion_evaluation(artifact_path)

    assert result is None
    assert error is not None
    assert "valid ground truth and ensemble verdict" in error


def test_model_export_excludes_synthetic_and_unsupported_metrics(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    model_dir = artifacts_dir / "models" / "sample"
    model_dir.mkdir(parents=True)
    (model_dir / "training_summary.json").write_text(json.dumps({
        "is_synthetic": True,
        "test_metrics": {"accuracy": 1.0, "split": "test"},
    }), encoding="utf-8")

    rows, errors = collect_model_evaluations(artifacts_dir)

    assert rows == []
    assert errors == []
