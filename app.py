#!/usr/bin/env python3
"""
RobustDeep Research Dashboard (Streamlit).
All research metrics load from artifacts/experiment_logs or are labelled SYNTHETIC.
"""
from __future__ import annotations

import json
import sys
import csv
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="RobustDeep Research Dashboard", layout="wide", page_icon="🔬")

DISCLAIMER = (
    "⚠️ **RESEARCH SYSTEM** — Metrics shown from real experiments are loaded from saved "
    "artifacts only. Any synthetic/smoke-test numbers are explicitly labelled "
    "**SYNTHETIC – NOT A RESEARCH RESULT**. Never treat demo predictions as peer-reviewed evidence."
)


def load_experiment_summaries():
    log_dir = ROOT / "artifacts" / "experiment_logs"
    ckpt_dir = ROOT / "artifacts" / "models"
    results = []
    if ckpt_dir.exists():
        for exp in ckpt_dir.iterdir():
            summary = exp / "training_summary.json"
            if summary.exists():
                with open(summary) as f:
                    data = json.load(f)
                data["path"] = str(exp)
                results.append(data)
    return results


def load_latest_artifact(filename: str):
    log_dir = ROOT / "artifacts" / "experiment_logs"
    candidates = sorted(log_dir.glob(f"*/{filename}"), key=lambda path: path.parent.name, reverse=True) if log_dir.exists() else []
    if not candidates:
        return None
    path = candidates[0]
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def page_home():
    st.title("RobustDeep")
    st.subheader("Explainable Vision Transformer-Based Video Deepfake Detection Under Cross-Dataset Distribution Shift")
    st.markdown(DISCLAIMER)
    st.markdown("""
    ### Research Objectives
    - Measure **cross-dataset generalization** (6-pair matrix, zero target tuning)
    - Compare CNN (EfficientNet) vs ViT under distribution shift
    - Ablate augmentation, loss, and backbone capacity
    - Provide Grad-CAM / Attention Rollout explanations + faithfulness checks
    - Stress-test against compression and blur

    ### Quick Navigation
    Use the sidebar to explore experiment logs, cross-dataset matrix, ablations, explainability, and dataset status.
    """)
    st.info("See `docs/REAL_DATA_QUICKSTART.md` for the 4-command real-data path.")


def page_experiments():
    st.header("Experiment Logs")
    st.markdown(DISCLAIMER)
    results = load_experiment_summaries()
    real_results = [result for result in results if "SYNTHETIC" not in str(result.get("label", "")) and not result.get("is_synthetic", False)]
    if not real_results:
        st.warning("No real experiment results yet - follow docs/REAL_DATA_QUICKSTART.md")
        smoke_dir = ROOT / "artifacts" / "checkpoints"
        smoke_results = []
        if smoke_dir.exists():
            for exp in smoke_dir.iterdir():
                summary = exp / "training_summary.json"
                if summary.exists():
                    smoke_results.append(json.loads(summary.read_text(encoding="utf-8")))
        if smoke_results:
            with st.expander("Smoke Test (non-research)"):
                st.json(smoke_results)
        return
    for r in real_results:
        label = r.get("label", "UNKNOWN")
        color = "red" if "SYNTHETIC" in str(label) else "green"
        st.markdown(f"### `{r.get('experiment_id')}`")
        st.markdown(f"**Label:** :{color}[{label}]")
        cols = st.columns(4)
        cols[0].metric("Best Val AUROC", f"{r.get('best_val_auroc', float('nan')):.4f}")
        cols[1].metric("Epochs", r.get("epochs_ran", "–"))
        cols[2].metric("Elapsed (s)", f"{r.get('elapsed_seconds', 0):.1f}")
        cols[3].metric("Synthetic?", str(r.get("is_synthetic", False)))
        with st.expander("Full history"):
            st.json(r.get("history", [])[-5:])


def page_smoke_test():
    st.header("Smoke Test (non-research)")
    smoke_dir = ROOT / "artifacts" / "checkpoints"
    summaries = []
    if smoke_dir.exists():
        for exp in smoke_dir.iterdir():
            summary = exp / "training_summary.json"
            if summary.exists():
                summaries.append(json.loads(summary.read_text(encoding="utf-8")))
    if not summaries:
        st.info("No synthetic smoke-test artifacts found.")
        return
    st.warning("Synthetic outputs are for pipeline checks only and are not research results.")
    st.json(summaries)


def page_cross_dataset():
    st.header("Cross-Dataset Matrix (6-pair)")
    st.markdown(DISCLAIMER)
    st.markdown("""
    Protocol: train on source → evaluate on target with **zero** target fine-tuning and **zero** threshold tuning.
    """)
    matrix = load_latest_artifact("cross_dataset_matrix.json")
    if not matrix or not matrix.get("rows"):
        st.warning("No real experiment results yet - follow docs/REAL_DATA_QUICKSTART.md")
        return
    st.dataframe(matrix["rows"], use_container_width=True)
    st.caption(matrix.get("definition", "Generalization gap is source test AUROC minus target test AUROC."))


def page_ablations():
    st.header("Ablations")
    results = load_latest_artifact("ablation_results.json")
    if not results or not results.get("rows"):
        st.warning("No real experiment results yet - follow docs/REAL_DATA_QUICKSTART.md")
        return
    st.dataframe(results["rows"], use_container_width=True)


def page_dataset_status():
    st.header("Dataset Status")
    st.markdown(DISCLAIMER)
    from src.utils.config import get_data_config, PROJECT_ROOT
    cfg = get_data_config()
    faces_root = PROJECT_ROOT / cfg["paths"]["faces_root"]
    for ds in ["faceforensics", "celebdf", "dfdc"]:
        st.subheader(ds)
        for split in ("train", "val", "test"):
            p = faces_root / ds / split
            n = len(list(p.glob("*"))) if p.exists() else 0
            st.write(f"  {split}: **{n}** video folders" + (" ✅" if n > 0 else " ❌ (run preprocess_faces.py)"))


def page_about():
    st.header("About & Limitations")
    st.markdown(DISCLAIMER)
    st.markdown("""
    - Primary reference: Petmezas (2026). We do **not** claim novelty for ViT or cross-dataset evaluation.
    - See `docs/research_gaps.md` for explicit differences and hypotheses.
    - Real datasets require user-provided videos (license restrictions). Code raises clear errors if faces are missing.
    - Only FaceForensics++, Celeb-DF, and DFDC are accepted by research entry points.
    """)


PAGES = {
    "Home": page_home,
    "Experiment Logs": page_experiments,
    "Smoke Test (non-research)": page_smoke_test,
    "Cross-Dataset Matrix": page_cross_dataset,
    "Ablations": page_ablations,
    "Dataset Status": page_dataset_status,
    "About & Limitations": page_about,
}


def main():
    st.sidebar.title("RobustDeep")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown(DISCLAIMER)
    PAGES[choice]()


if __name__ == "__main__":
    main()
