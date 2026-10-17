"""Temporary ML core test — run and delete."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Checking PyTorch...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
except ImportError:
    print("  PyTorch not installed — ML tests skipped.")
    sys.exit(0)

try:
    import timm
    HAS_TIMM = True
    print(f"  timm: {timm.__version__}")
except ImportError:
    HAS_TIMM = False
    print("  timm not installed")

# ── CNN model ─────────────────────────────────────────────────────────────────
print("\n[1] CNN model (EfficientNet-B0)...")
from src.utils.config import ModelConfig
from src.models.factory import create_model, get_model_size

cfg = ModelConfig(type="cnn", backbone="efficientnet_b0", pretrained=False)
model = create_model(cfg, "cpu")
info = model.get_model_info()
print(f"  Parameters: {info['total_parameters']:,}")

x = torch.randn(2, 3, 224, 224)
out = model(x)
assert out.shape == (2, 2), f"Expected (2,2) got {out.shape}"
print(f"  Forward pass: {out.shape} — OK")

# ── ViT model ─────────────────────────────────────────────────────────────────
if HAS_TIMM:
    print("\n[2] ViT model (ViT-Tiny)...")
    cfg_vit = ModelConfig(type="vit", backbone="vit_tiny_patch16_224", pretrained=False)
    model_vit = create_model(cfg_vit, "cpu")
    out_vit = model_vit(x)
    assert out_vit.shape == (2, 2), f"Expected (2,2) got {out_vit.shape}"
    print(f"  Forward pass: {out_vit.shape} — OK")

# ── Training step ──────────────────────────────────────────────────────────────
print("\n[3] Training step...")
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.training.trainer import Trainer
from src.utils.config import TrainingConfig
from src.utils.reproducibility import set_seed

set_seed(42)
train_ds = SyntheticDataset(num_samples=32, seed=42, split="train")
val_ds   = SyntheticDataset(num_samples=8,  seed=43, split="val")
test_ds  = SyntheticDataset(num_samples=8,  seed=44, split="test")
loaders = create_dataloaders(train_ds, val_ds, test_ds, batch_size=8, num_workers=0)

cfg_train = TrainingConfig()
cfg_train.epochs = 1
cfg_train.early_stopping_enabled = False
cfg_train.log_interval = 10

cfg2 = ModelConfig(type="cnn", backbone="efficientnet_b0", pretrained=False)
model2 = create_model(cfg2, "cpu")

trainer = Trainer(model2, cfg_train, "cpu", "test_exp_001")
result = trainer.train(loaders["train"], loaders["val"], num_epochs=1)
print(f"  Training result keys: {list(result.keys())}")
assert "best_val_metric" in result or "checkpoint_dir" in result or "total_time_s" in result
print("  Training step: OK")

# ── Evaluation step ────────────────────────────────────────────────────────────
print("\n[4] Evaluation step...")
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model2, "cpu")
metrics = evaluator.evaluate(loaders["test"], dataset_name="synthetic_test")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  AUROC:    {metrics['auroc']:.4f}")
assert 0 <= metrics["accuracy"] <= 1
assert 0 <= metrics["auroc"] <= 1
print("  Evaluation step: OK")

# ── Robustness test ────────────────────────────────────────────────────────────
print("\n[5] Robustness tester...")
from src.robustness.tester import RobustnessTester

tester = RobustnessTester(model2, "cpu")
stress_configs = [
    {"name": "jpeg_q50", "transform": "jpeg_compression", "params": {"quality": 50}},
    {"name": "blur_k5",  "transform": "gaussian_blur",    "params": {"kernel_size": 5, "sigma": 1.0}},
]
rob_results = tester.run_stress_tests(
    loaders["test"],
    stress_configs=stress_configs,
    clean_metrics=metrics,
)
assert "jpeg_q50" in rob_results
assert "blur_k5" in rob_results
print(f"  jpeg_q50 AUROC: {rob_results['jpeg_q50'].get('auroc', 0):.4f}")
print(f"  blur_k5  AUROC: {rob_results['blur_k5'].get('auroc', 0):.4f}")
print("  Robustness step: OK")

print()
print("=" * 50)
print("ML CORE: ALL 5 STEPS PASSED")
print("=" * 50)
