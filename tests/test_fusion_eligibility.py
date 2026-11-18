from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ensemble.fusion import ModelResult, _fuse_model_results


@pytest.fixture
def registry():
    return SimpleNamespace(
        ensemble_threshold=0.5,
        uncertain_low=0.4,
        uncertain_high=0.6,
        disagreement_std_threshold=0.2,
    )


def model_result(model_id: str, probability: float, included: bool, weight: float):
    return ModelResult(
        model_id=model_id,
        display_name=model_id,
        dataset="test",
        architecture="test",
        probability=probability,
        weight=weight,
        val_auc=0.8 if included else 0.4,
        included=included,
    )


def test_excluded_models_do_not_affect_fusion_or_disagreement(registry):
    results = [
        model_result("eligible_a", 0.8, included=True, weight=1.0),
        model_result("eligible_b", 0.85, included=True, weight=1.0),
        model_result("excluded", 0.0, included=False, weight=0.0),
    ]

    fused, verdict, confidence, disagreement = _fuse_model_results(results, registry)

    assert fused == pytest.approx(0.825)
    assert verdict == "FAKE"
    assert confidence == pytest.approx(82.5)
    assert disagreement == pytest.approx(0.025)


def test_no_eligible_models_produces_uncertain_without_fusing_excluded_scores(registry):
    results = [model_result("excluded", 0.99, included=False, weight=0.0)]

    assert _fuse_model_results(results, registry) == (0.5, "UNCERTAIN", 0.0, 0.0)


def test_large_disagreement_produces_uncertain_even_outside_score_band(registry):
    results = [
        model_result("eligible_a", 0.95, included=True, weight=1.0),
        model_result("eligible_b", 0.4, included=True, weight=1.0),
    ]

    fused, verdict, _, disagreement = _fuse_model_results(results, registry)

    assert fused > registry.uncertain_high
    assert disagreement > registry.disagreement_std_threshold
    assert verdict == "UNCERTAIN"
