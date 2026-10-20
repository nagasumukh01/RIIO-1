"""
Verify load_experiment_data returns all fields populated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import load_experiment_data

data = load_experiment_data()

print("--- DATA SUMMARY ---")
print("Total experiments:", len(data.get("experiments", [])))
print("Comparison rows:", len(data.get("comparison", [])))
print("Results keys:", list(data["results"].keys()) if data.get("results") else None)
print("Cross-dataset keys:", list(data["cross_dataset"].keys()) if data.get("cross_dataset") else None)
print("Robustness keys:", list(data["robustness"].keys()) if data.get("robustness") else None)

assert len(data.get("experiments", [])) >= 16, "Expected at least 16 experiments"
assert data.get("results") is not None, "Expected results not None"
assert "augmentation_ablation" in data["results"], "Expected augmentation_ablation"
assert "loss_ablation" in data["results"], "Expected loss_ablation"
assert "backbone_ablation" in data["results"], "Expected backbone_ablation"
assert data.get("cross_dataset") is not None, "Expected cross_dataset not None"
assert data.get("robustness") is not None, "Expected robustness not None"

print("\nSUCCESS: All dashboard data validated!")
