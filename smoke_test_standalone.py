"""
Standalone smoke test runner.
Tests the core logic of RobustDeep using ONLY numpy / pyyaml (no torch needed).
Run with: python smoke_test_standalone.py
"""

import sys
import os
import json
import time

# Fix Windows cp1252 console by forcing UTF-8
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = []


def test(name, fn):
    """Run one test, catch all exceptions, print result."""
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        RESULTS.append(("PASS", name, elapsed, None))
        print(f"  [PASS] {name:<58}  ({elapsed*1000:.0f}ms)")
    except Exception as exc:
        elapsed = time.time() - start
        RESULTS.append(("FAIL", name, elapsed, str(exc)))
        print(f"  [FAIL] {name:<58}  {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config_tests():
    print("\n[1] Configuration System")

    def yaml_loading():
        import yaml
        with open("configs/data.yaml") as f:
            d = yaml.safe_load(f)
        assert "data" in d
        assert d["data"]["face_size"] == 224

    def data_config():
        from src.utils.config import DataConfig
        cfg = DataConfig.from_dict({"face_size": 112, "frames_per_video": 8})
        assert cfg.face_size == 112

    def model_config():
        from src.utils.config import ModelConfig
        cfg = ModelConfig()
        assert cfg.num_classes == 2

    def training_config():
        from src.utils.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.seed == 42

    def split_ratios():
        from src.utils.config import DataConfig
        cfg = DataConfig()
        total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios sum to {total}"

    def load_all():
        from src.utils.config import load_all_configs
        cfg = load_all_configs("configs")
        assert isinstance(cfg, dict) and "data" in cfg

    def ignores_unknown_keys():
        from src.utils.config import DataConfig
        cfg = DataConfig.from_dict({"face_size": 100, "unknown_key_xyz": True})
        assert cfg.face_size == 100

    test("YAML loading - data.yaml", yaml_loading)
    test("DataConfig.from_dict()", data_config)
    test("ModelConfig defaults", model_config)
    test("TrainingConfig defaults", training_config)
    test("Split ratios sum to 1.0", split_ratios)
    test("load_all_configs()", load_all)
    test("Config ignores unknown keys", ignores_unknown_keys)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────────────────────────────────────
def run_metrics_tests():
    print("\n[2] Metrics Computation")
    import numpy as np

    def perfect():
        from src.utils.metrics import compute_all_metrics
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_all_metrics(labels, preds, scores)
        assert m["accuracy"] == 1.0, f"acc={m['accuracy']}"
        assert m["auroc"] == 1.0,   f"auroc={m['auroc']}"
        assert m["tp"] == 2 and m["tn"] == 2

    def mixed():
        from src.utils.metrics import compute_all_metrics
        labels = np.array([0, 1, 0, 1, 0, 1])
        preds  = np.array([0, 1, 1, 0, 0, 1])
        scores = np.array([0.3, 0.7, 0.6, 0.4, 0.2, 0.8])
        m = compute_all_metrics(labels, preds, scores)
        assert 0 <= m["accuracy"] <= 1
        assert m["total_samples"] == 6
        assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 6

    def cross_drop():
        from src.utils.metrics import compute_cross_dataset_drop
        drop = compute_cross_dataset_drop({"auroc": 0.95}, {"auroc": 0.70}, "auroc")
        assert drop > 20, f"Expected >20%, got {drop:.1f}%"

    def rob_drop():
        from src.utils.metrics import compute_robustness_drop
        drop = compute_robustness_drop({"auroc": 0.90}, {"auroc": 0.65}, "auroc")
        assert drop > 20

    def zero_baseline():
        from src.utils.metrics import compute_cross_dataset_drop
        drop = compute_cross_dataset_drop({"auroc": 0.0}, {"auroc": 0.5}, "auroc")
        assert drop == 0.0, "Zero baseline should not divide"

    def eer_range():
        from src.utils.metrics import compute_eer
        labels = np.array([0]*50 + [1]*50)
        rng = np.random.RandomState(7)
        scores = np.concatenate([rng.uniform(0, 0.5, 50), rng.uniform(0.5, 1, 50)])
        eer, _ = compute_eer(labels, scores)
        assert 0 <= eer <= 1

    def cm_shape():
        from src.utils.metrics import compute_all_metrics
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([0, 1, 1, 0])
        scores = np.array([0.2, 0.6, 0.8, 0.4])
        m = compute_all_metrics(labels, preds, scores)
        cm = m["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    test("Perfect predictions -> acc=auroc=1.0", perfect)
    test("Mixed predictions -> valid ranges", mixed)
    test("Cross-dataset drop computation", cross_drop)
    test("Robustness drop computation", rob_drop)
    test("Zero baseline -> no division error", zero_baseline)
    test("EER in [0,1]", eer_range)
    test("Confusion matrix is 2x2", cm_shape)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def run_augmentation_tests():
    print("\n[3] Augmentation Pipeline")
    import numpy as np

    def img():
        return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    def jpeg():
        from src.data.augmentation import JpegCompression
        r = JpegCompression((30, 90))(image=img())
        assert r["image"].shape == (224, 224, 3)

    def blur():
        from src.data.augmentation import GaussianBlur
        r = GaussianBlur((3, 5))(image=img())
        assert r["image"].shape == (224, 224, 3)

    def resize():
        from src.data.augmentation import ResizeReencode
        r = ResizeReencode((0.5, 0.9))(image=img())
        assert r["image"].shape == (224, 224, 3)

    def jitter():
        from src.data.augmentation import ColorJitter
        r = ColorJitter(brightness=0.2)(image=img())
        assert r["image"].shape == (224, 224, 3)

    def flip():
        from src.data.augmentation import HorizontalFlip
        src = img()
        r = HorizontalFlip(probability=1.0)(image=src)
        assert np.array_equal(r["image"], np.fliplr(src))

    def eval_no_aug():
        from src.data.augmentation import build_augmentation_pipeline
        config = {
            "jpeg_compression": {"enabled": True, "quality_range": [30, 95]},
            "gaussian_blur": {"enabled": True, "kernel_range": [3, 7]},
        }
        pipeline = build_augmentation_pipeline(config, is_training=False)
        assert len(pipeline.transforms) == 0, "Eval mode must NOT apply augmentation!"

    def compose():
        from src.data.augmentation import ComposeTransforms, GaussianBlur, HorizontalFlip
        t = ComposeTransforms([GaussianBlur(), HorizontalFlip()])
        r = t(image=img())
        assert r["image"].shape == (224, 224, 3)

    def rob_transform():
        from src.data.augmentation import build_robustness_transform
        t = build_robustness_transform("jpeg_compression", {"quality": 50})
        r = t(image=img())
        assert r["image"].shape == (224, 224, 3)

    def combined_rob():
        from src.data.augmentation import build_robustness_transform
        t = build_robustness_transform("combined", {"jpeg_quality": 50, "blur_kernel": 5, "resize_scale": 0.7})
        r = t(image=img())
        assert r["image"].shape == (224, 224, 3)

    test("JPEG compression output shape", jpeg)
    test("Gaussian blur output shape", blur)
    test("Resize re-encode output shape", resize)
    test("Color jitter output shape", jitter)
    test("Horizontal flip (deterministic)", flip)
    test("Eval mode -> zero augmentations applied", eval_no_aug)
    test("ComposeTransforms chains correctly", compose)
    test("Robustness transform: jpeg", rob_transform)
    test("Robustness transform: combined stress", combined_rob)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Experiment Tracker
# ─────────────────────────────────────────────────────────────────────────────
def run_tracker_tests():
    print("\n[4] Experiment Tracker (SQLite)")
    import tempfile

    tmpdir = tempfile.mkdtemp()
    db = os.path.join(tmpdir, "test.db")

    def mk():
        from src.experiments.tracker import ExperimentTracker
        return ExperimentTracker(db)

    def create():
        t = mk()
        eid = t.create_experiment("t1", "smoke_test", "EffNet", "cnn", "syn")
        assert eid and len(eid) > 5

    def get():
        t = mk()
        eid = t.create_experiment("t2", "training", "ViT", "vit", "ff++")
        exp = t.get_experiment(eid)
        assert exp is not None and exp["name"] == "t2"

    def log_metrics():
        t = mk()
        eid = t.create_experiment("t3", "training", "EN", "cnn", "ff++")
        t.log_metrics(eid, {"accuracy": 0.85, "auroc": 0.92})
        metrics = t.get_metrics(eid)
        names = [m["metric_name"] for m in metrics]
        assert "accuracy" in names and "auroc" in names

    def complete():
        t = mk()
        eid = t.create_experiment("t4", "training", "EN", "cnn", "ff++")
        t.complete_experiment(eid, status="completed", duration_s=42.0)
        exp = t.get_experiment(eid)
        assert exp["status"] == "completed"
        assert exp["duration_s"] == 42.0

    def list_exps():
        t = mk()
        before = len(t.get_all_experiments())
        t.create_experiment("l1", "training", "m", "cnn", "s")
        t.create_experiment("l2", "training", "m", "cnn", "s")
        assert len(t.get_all_experiments()) == before + 2

    def nonexistent():
        t = mk()
        assert t.get_experiment("does_not_exist_xyz") is None

    def filter_type():
        t = mk()
        t.create_experiment("f1", "training", "m", "cnn", "s")
        t.create_experiment("f2", "smoke_test", "m", "cnn", "s")
        tr = t.get_all_experiments("training")
        sm = t.get_all_experiments("smoke_test")
        assert all(e["type"] == "training" for e in tr)
        assert all(e["type"] == "smoke_test" for e in sm)

    def export():
        t = mk()
        t.export_results(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "experiments.json"))

    test("Create experiment -> valid ID", create)
    test("Get experiment by ID", get)
    test("Log and retrieve metrics", log_metrics)
    test("Complete experiment -> status updated", complete)
    test("List all experiments", list_exps)
    test("Nonexistent ID -> None", nonexistent)
    test("Filter experiments by type", filter_type)
    test("Export results to JSON", export)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reproducibility & Logging
# ─────────────────────────────────────────────────────────────────────────────
def run_reproducibility_tests():
    print("\n[5] Reproducibility & Logging")

    def hardware_info():
        from src.utils.reproducibility import get_hardware_info
        info = get_hardware_info()
        assert "python_version" in info
        assert "torch_version" in info

    def logger_setup():
        from src.utils.logging_utils import setup_logger, get_logger
        lg = setup_logger("test_check_logger")
        assert lg is not None
        lg2 = get_logger("test_check_logger")
        assert lg2 is not None

    test("Hardware info fields present", hardware_info)
    test("Logger setup and retrieval", logger_setup)


# ─────────────────────────────────────────────────────────────────────────────
# 6. YAML Config Integrity
# ─────────────────────────────────────────────────────────────────────────────
def run_yaml_integrity_tests():
    print("\n[6] YAML Config Integrity")
    import yaml

    def data_yaml():
        with open("configs/data.yaml") as f:
            d = yaml.safe_load(f)
        assert d["data"]["face_size"] == 224
        assert d["data"]["frames_per_video"] > 0
        aug = d.get("augmentation", {})
        assert "jpeg_compression" in aug
        assert "gaussian_blur" in aug

    def models_yaml():
        with open("configs/models.yaml") as f:
            d = yaml.safe_load(f)
        models = d.get("models", {})
        assert "efficientnet_b0" in models
        assert "vit_base" in models

    def training_yaml():
        with open("configs/training.yaml") as f:
            d = yaml.safe_load(f)
        tr = d.get("training", {})
        assert tr.get("epochs", 0) > 0
        assert tr.get("seed") == 42

    def experiments_yaml():
        with open("configs/experiments.yaml") as f:
            d = yaml.safe_load(f)
        pairs = d["experiments"]["cross_dataset_pairs"]
        assert len(pairs) == 6, f"Expected 6 pairs, got {len(pairs)}"
        for p in pairs:
            assert "train" in p and "test" in p

    test("data.yaml: face_size=224, augments defined", data_yaml)
    test("models.yaml: efficientnet_b0 and vit_base exist", models_yaml)
    test("training.yaml: epochs > 0, seed = 42", training_yaml)
    test("experiments.yaml: exactly 6 cross-dataset pairs", experiments_yaml)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Project Structure
# ─────────────────────────────────────────────────────────────────────────────
def run_structure_tests():
    print("\n[7] Project File Structure")

    required = [
        "app.py", "requirements.txt", "README.md", "setup.py",
        "configs/data.yaml", "configs/models.yaml",
        "configs/training.yaml", "configs/experiments.yaml",
        "src/__init__.py",
        "src/data/dataset.py", "src/data/preprocessing.py",
        "src/data/augmentation.py", "src/data/real_dataset.py",
        "src/models/cnn_baseline.py", "src/models/vit_detector.py", "src/models/factory.py",
        "src/training/trainer.py", "src/training/losses.py",
        "src/evaluation/evaluator.py",
        "src/experiments/tracker.py", "src/experiments/runner.py",
        "src/explainability/explainer.py",
        "src/robustness/tester.py",
        "src/api/__init__.py", "src/api/app.py",
        "src/utils/config.py", "src/utils/metrics.py",
        "src/utils/logging_utils.py", "src/utils/reproducibility.py",
        "scripts/train.py", "scripts/evaluate.py", "scripts/run_ablations.py",
        "scripts/run_cross_dataset.py", "scripts/run_all.py", "scripts/prepare_data.py",
        "scripts/infer.py", "scripts/preprocess_faces.py",
        "tests/test_metrics.py", "tests/test_data.py",
        "tests/test_models.py", "tests/test_config_and_tracker.py",
        "tests/test_integration.py",
        "docs/literature_review.md", "docs/research_gaps.md",
        "docs/dataset_protocol.md", "docs/architecture.md",
        "docs/experiment_protocol.md", "docs/limitations.md",
        "docs/results_template.md",
        ".github/workflows/ci.yml",
    ]

    missing = [f for f in required if not os.path.exists(f)]

    def all_files_present():
        assert not missing, f"Missing {len(missing)} files: {missing[:5]}..."

    def artifacts_dirs():
        for d in ["artifacts/checkpoints", "artifacts/experiment_logs", "artifacts/figures",
                  "artifacts/cache"]:
            os.makedirs(d, exist_ok=True)
        for d in ["artifacts/checkpoints", "artifacts/experiment_logs"]:
            assert os.path.isdir(d)

    test(f"All {len(required)} required source files present", all_files_present)
    test("Artifact directories creatable", artifacts_dirs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 74)
    print("  RobustDeep -- Standalone Smoke Test (numpy/pyyaml only)")
    print("=" * 74)

    run_config_tests()
    run_metrics_tests()
    run_augmentation_tests()
    run_tracker_tests()
    run_reproducibility_tests()
    run_yaml_integrity_tests()
    run_structure_tests()

    passed = [r for r in RESULTS if r[0] == "PASS"]
    failed = [r for r in RESULTS if r[0] == "FAIL"]

    print("\n" + "=" * 74)
    print(f"  RESULT: {len(passed)} passed, {len(failed)} failed / {len(RESULTS)} total")
    print("=" * 74)

    if failed:
        print("\n  FAILED:")
        for _, name, _, err in failed:
            print(f"    FAIL: {name}")
            print(f"      ->  {err}")

    # Persist results
    os.makedirs("artifacts/experiment_logs", exist_ok=True)
    out = {
        "passed": len(passed), "failed": len(failed), "total": len(RESULTS),
        "tests": [
            {"status": r[0], "name": r[1], "duration_ms": round(r[2]*1000, 1), "error": r[3]}
            for r in RESULTS
        ]
    }
    out_path = "artifacts/experiment_logs/smoke_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results -> {out_path}")
    print("=" * 74)

    if failed:
        sys.exit(1)
    print("  ALL TESTS PASSED -- Pipeline infrastructure is valid!")
    sys.exit(0)


if __name__ == "__main__":
    main()
