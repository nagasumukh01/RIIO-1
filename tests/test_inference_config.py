from __future__ import annotations

import numpy as np
import pytest
import torch

from src.api.server import (
    MODEL_INFERENCE_CONFIG,
    MODEL_REGISTRY,
    _fake_probability,
    _inference_config,
    _is_fake,
    _model_input,
    _normalise,
)


MODEL_IDS = [model_id for model_id in MODEL_REGISTRY if model_id != "cross_ensemble"]


def test_each_registered_model_has_its_own_inference_config():
    assert set(MODEL_INFERENCE_CONFIG) == set(MODEL_REGISTRY)
    assert len({id(_inference_config(model_id)) for model_id in MODEL_REGISTRY}) == len(MODEL_REGISTRY)
    for model_id in MODEL_REGISTRY:
        config = _inference_config(model_id)
        assert config["input_size"] == 224
        assert config["mean"] == [0.485, 0.456, 0.406]
        assert config["std"] == [0.229, 0.224, 0.225]
        assert config["num_classes"] == 2
        assert config["output_units"] == 1
        assert config["real_index"] == 0
        assert config["fake_index"] == 1
        assert config["threshold"] == 0.5


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_preprocessing_returns_normalized_chw_tensor(model_id):
    image = np.full((32, 48, 3), 128, dtype=np.uint8)
    tensor = _normalise(image, model_id)

    assert tensor.shape == (3, 224, 224)
    expected = (128 / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    torch.testing.assert_close(tensor[:, 100, 100], expected)


def test_fake_logit_and_threshold_use_model_config(monkeypatch):
    config = dict(MODEL_INFERENCE_CONFIG["celebdf_b0"])
    config["fake_index"] = 0
    config["threshold"] = 0.7
    monkeypatch.setitem(MODEL_INFERENCE_CONFIG, "celebdf_b0", config)

    assert _fake_probability(torch.tensor([2.0]), "celebdf_b0") < 0.2
    assert _fake_probability(torch.tensor([-2.0]), "celebdf_b0") > 0.8
    assert _is_fake(0.7, "celebdf_b0")
    assert not _is_fake(0.69, "celebdf_b0")


def test_image_input_uses_primary_view_and_video_uses_configured_view_count():
    snapshots = [{"crop": np.full((16, 16, 3), index * 30, dtype=np.uint8)} for index in range(6)]

    image_input = _model_input(snapshots, "celebdf_b0", is_video=False)
    video_input = _model_input(snapshots, "ff_vit", is_video=True)

    assert image_input.shape == (1, 1, 3, 224, 224)
    assert video_input.shape == (1, 4, 3, 224, 224)
