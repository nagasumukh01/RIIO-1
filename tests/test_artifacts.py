from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.evaluation.artifacts import export_evaluation_artifacts


def test_evaluation_artifacts_have_research_structure(tmp_path: Path):
    result = export_evaluation_artifacts(
        torch.tensor([-2.0, 2.0]),
        torch.tensor([0.0, 1.0]),
        ["real_001", "fake_001"],
        "test",
        "faceforensics",
        tmp_path,
    )
    assert Path(result["predictions"]).exists()
    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0].keys() == {"video_id", "label", "score", "split"}
    assert (tmp_path / "roc_test.json").exists()
    assert (tmp_path / "roc_test.png").exists()
    assert (tmp_path / "confusion_matrix_test.json").exists()
    assert (tmp_path / "confusion_matrix_test.png").exists()
    assert json.loads((tmp_path / "roc_test.json").read_text())["metrics"]["is_synthetic"] is False
