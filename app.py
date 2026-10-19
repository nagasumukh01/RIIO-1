"""
RobustDeep Research Dashboard - Streamlit Application
A comprehensive research dashboard for deepfake detection experiments.

Run with: streamlit run app.py
"""

import sys
import os
import json
import time
import io
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ─── Page Configuration ───
st.set_page_config(
    page_title="RobustDeep — Deepfake Detection Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        color: #b0b8d1;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .disclaimer-box {
        background: linear-gradient(145deg, #2d1b00, #1a1100);
        border: 1px solid #ff9800;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #ffb74d;
        font-size: 0.9rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-completed { background: #1b5e20; color: #81c784; }
    .badge-running { background: #e65100; color: #ffb74d; }
    .badge-pending { background: #263238; color: #90a4ae; }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0e0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e, #16213e);
    }

    div[data-testid="stSidebar"] .stRadio > label {
        color: #b0b8d1;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    .research-note {
        background: linear-gradient(145deg, #0d1b2a, #1b263b);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #b0b8d1;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ───
def load_experiment_data():
    """Load all available experiment data."""
    data = {
        'experiments': [],
        'metrics': {},
        'results': None,
        'cross_dataset': None,
        'robustness': None,
    }

    # Load from SQLite tracker
    try:
        from src.experiments.tracker import ExperimentTracker
        tracker = ExperimentTracker()
        data['experiments'] = tracker.get_all_experiments()
        data['comparison'] = tracker.get_comparison_table()
    except Exception:
        data['comparison'] = []

    # Load all_results.json
    results_path = Path("artifacts/experiment_logs/all_results.json")
    if results_path.exists():
        with open(results_path) as f:
            data['results'] = json.load(f)

    # Load cross-dataset matrix
    matrix_path = Path("artifacts/experiment_logs/cross_dataset_matrix.json")
    if matrix_path.exists():
        with open(matrix_path) as f:
            data['cross_dataset'] = json.load(f)

    # Load robustness results
    for p in Path("artifacts/experiment_logs").glob("robustness_*.json"):
        with open(p) as f:
            data['robustness'] = json.load(f)

    return data


def create_plotly_theme():
    """Common Plotly theme for consistent charts."""
    return {
        'paper_bgcolor': 'rgba(15,12,41,0.8)',
        'plot_bgcolor': 'rgba(26,26,46,0.8)',
        'font': {'color': '#b0b8d1', 'family': 'Inter'},
        'title_font': {'size': 16, 'color': '#e0e0e0'},
        'xaxis': {'gridcolor': 'rgba(102,126,234,0.1)', 'zerolinecolor': 'rgba(102,126,234,0.2)'},
        'yaxis': {'gridcolor': 'rgba(102,126,234,0.1)', 'zerolinecolor': 'rgba(102,126,234,0.2)'},
    }


def metric_card(label, value, suffix="", delta=None):
    """Create a styled metric card."""
    delta_html = ""
    if delta is not None:
        color = "#81c784" if delta >= 0 else "#e57373"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">{arrow} {abs(delta):.1f}%</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}{suffix}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar Navigation ───
with st.sidebar:
    st.markdown("### 🔬 RobustDeep")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Research Overview",
            "🎬 Video Analysis & Inference",
            "📊 Experiment Results",
            "🔄 Cross-Dataset Matrix",
            "🏗️ CNN vs ViT Comparison",
            "🧪 Ablation Studies",
            "🎯 Explainability",
            "🛡️ Robustness Tests",
            "📈 Training History",
            "🔍 Run Experiments",
            "📋 Dataset & Protocol",
            "📝 Documentation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #666; padding: 0.5rem;">
        <strong>RobustDeep v1.0</strong><br>
        Research-grade deepfake detection<br>
        under cross-dataset distribution shift
    </div>
    """, unsafe_allow_html=True)


# ─── Load Data ───
data = load_experiment_data()


# ═══════════════════════════════════════════════════════════════
# PAGE: Research Overview
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Research Overview":
    st.markdown("""
    <div class="main-header">
        <h1>RobustDeep: Explainable ViT-Based Video Deepfake Detection</h1>
        <p>Under Cross-Dataset Distribution Shift — Research Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>DISCLAIMER:</strong> This is a probabilistic research system and must not be
        treated as conclusive forensic evidence. All metrics shown are from actual model predictions
        or are explicitly labeled as illustrative examples.
    </div>
    """, unsafe_allow_html=True)

    # Research overview
    col1, col2, col3, col4 = st.columns(4)

    num_experiments = len(data.get('experiments', []))
    completed = len([e for e in data.get('experiments', []) if e.get('status') == 'completed'])

    with col1:
        metric_card("Total Experiments", str(num_experiments))
    with col2:
        metric_card("Completed", str(completed))
    with col3:
        metric_card("Model Types", "3", suffix=" (CNN/ViT/Hybrid)")
    with col4:
        metric_card("Ablation Types", "3", suffix=" (Aug/Loss/BB)")

    st.markdown('<div class="section-header">Research Problem</div>', unsafe_allow_html=True)

    st.markdown("""
    Deepfake detectors often achieve high accuracy on data drawn from the same distribution
    as their training set, yet collapse when tested on videos from different datasets,
    manipulation methods, or compression settings.

    **This system investigates:**
    1. Whether Vision Transformers generalize better than CNN baselines under distribution shift
    2. How performance changes across the **6-pair cross-dataset evaluation matrix**
    3. Impact of **compression-aware training** on robustness
    4. Which augmentation strategies, loss functions, and backbone capacities influence cross-dataset performance
    5. Whether model explanations (Grad-CAM / attention maps) remain meaningful across domains
    """)

    st.markdown('<div class="section-header">Key Research Gaps Addressed</div>', unsafe_allow_html=True)

    gaps = {
        "Gap A": "6-pair cross-dataset evaluation with NO target fine-tuning",
        "Gap B": "Diffusion-generated manipulation evaluation",
        "Gap C": "Controlled augmentation ablation (6 strategies)",
        "Gap D": "Loss function ablation (CE vs Focal vs Weighted BCE)",
        "Gap E": "Backbone capacity ablation (4 architectures)",
        "Gap F": "Explainability under distribution shift + faithfulness checks",
        "Gap G": "Full reproducibility with saved configs, predictions, and seeds",
    }

    for gap, desc in gaps.items():
        st.markdown(f"**{gap}:** {desc}")

    st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    ```
    Video → Metadata Validation → Frame Sampling → Face Detection → Normalized Crops
         → Training Augmentation → Model Training/Inference → Frame-Level Scores
         → Video-Level Aggregation → Metrics + Uncertainty → Explanation Visualization
         → Saved Experiment Record
    ```
    """)

    # Reference paper
    st.markdown('<div class="section-header">Reference Paper</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="research-note">
        <strong>Petmezas (2026)</strong> — "Cross-dataset video deepfake detection using
        Transformer and CNN architectures", <em>Machine Vision and Applications</em>.<br>
        DOI: 10.1007/s00138-026-01809-w<br><br>
        This project extends the closest published benchmark by adding the explainability
        layer the paper itself flags as missing, testing against newer manipulation types,
        and using a more granular 6-pair cross-dataset matrix with stricter no-target-tuning rules.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Video Analysis & Inference
# ═══════════════════════════════════════════════════════════════
elif page == "🎬 Video Analysis & Inference":
    st.markdown("""
    <div class="main-header">
        <h1>Video Deepfake Detection & Explanation</h1>
        <p>End-to-end video inference with temporal aggregation and visual attribution maps</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>MANDATORY FORENSIC DISCLAIMER:</strong> This is a probabilistic research system and must NOT be
        treated as conclusive forensic evidence. Model confidence is a statistical probability based on training
        distributions and may degrade significantly under unseen manipulations, compression, or lighting shifts.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">1. Select or Upload Video</div>', unsafe_allow_html=True)

    # Search for available sample videos in data/sample_videos
    sample_videos_dir = Path("data/sample_videos")
    available_samples = list(sample_videos_dir.glob("*.mp4")) if sample_videos_dir.exists() else []

    source_mode = st.radio(
        "Video Source",
        ["Use Preloaded Sample Video", "Upload Your Own Video (.mp4 / .avi / .mov)"],
        horizontal=True,
    )

    selected_video_path = None
    temp_uploaded_path = None

    if source_mode == "Use Preloaded Sample Video":
        if available_samples:
            sample_choice = st.selectbox(
                "Choose a sample test video:",
                available_samples,
                format_func=lambda p: f"{p.name} ({'Real Sample' if 'real' in p.name else ('Diffusion Fake' if 'diffusion' in p.name else 'Face-Swap Fake')})",
            )
            selected_video_path = str(sample_choice)
        else:
            st.info("No sample videos found in data/sample_videos. Run python scripts/create_sample_videos.py")
    else:
        uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_dir = Path("data/temp_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_uploaded_path = temp_dir / f"uploaded_{int(time.time())}_{uploaded_file.name}"
            with open(temp_uploaded_path, "wb") as f:
                f.write(uploaded_file.read())
            selected_video_path = str(temp_uploaded_path)
            st.success(f"Uploaded successfully ({uploaded_file.size / (1024*1024):.2f} MB)")

    if selected_video_path and os.path.exists(selected_video_path):
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.markdown("#### Video Preview")
            st.video(selected_video_path)

        with v_col2:
            st.markdown("#### Model & Sampling Configuration")
            model_key = st.selectbox(
                "Detector Architecture",
                ["efficientnet_b0", "vit_base", "hybrid"],
                format_func=lambda x: {
                    "efficientnet_b0": "EfficientNet-B0 (CNN Baseline - 4.3M)",
                    "vit_base": "ViT-Base (Vision Transformer - 86.6M)",
                    "hybrid": "Hybrid CNN-ViT (14.8M)"
                }.get(x, x),
            )

            # Available checkpoints
            ckpt_dir = Path("artifacts/checkpoints")
            available_ckpts = list(ckpt_dir.glob("**/*.pt")) if ckpt_dir.exists() else []

            ckpt_options = ["None (Default Pretrained Weights)"] + [str(p) for p in available_ckpts]
            selected_ckpt_str = st.selectbox(
                "Checkpoint Path",
                ckpt_options,
                format_func=lambda x: "Default Pretrained Weights" if "None" in x else Path(x).parent.name + " / " + Path(x).name
            )
            checkpoint_path = None if "None" in selected_ckpt_str else selected_ckpt_str

            num_frames = st.slider("Frames to Sample from Video", min_value=5, max_value=30, value=15, step=5)
            generate_heatmaps = st.checkbox("Generate Visual Explanation Maps (Grad-CAM / Attention Rollout)", value=True)

        if st.button("🚀 Analyze Video for Deepfake Manipulation", type="primary", use_container_width=True):
            with st.spinner("Extracting frames, sampling faces, and running neural inference..."):
                try:
                    from scripts.infer import load_model, extract_video_frames, predict_image
                    import torch
                    import cv2
                    from src.explainability.explainer import GradCAM, AttentionRollout, create_heatmap_overlay

                    device = "cpu"
                    model, config = load_model(model_key, checkpoint_path=checkpoint_path, device=device)

                    frames = extract_video_frames(selected_video_path, num_frames=num_frames, face_size=224)

                    if not frames:
                        st.error("Could not extract any valid face frames from the video. Please verify the video codec and content.")
                    else:
                        frame_scores = []
                        analyzed_data = []

                        # Initialize explainer if requested
                        explainer = None
                        if generate_heatmaps:
                            if config.type == "cnn":
                                explainer = GradCAM(model)
                            elif config.type == "vit":
                                explainer = AttentionRollout(model)

                        for idx, frame_np in enumerate(frames):
                            # Preprocess frame
                            norm_img = frame_np.astype(np.float32) / 255.0
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            norm_img = (norm_img - mean) / std
                            tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float().unsqueeze(0)

                            # Predict
                            pred_res = predict_image(model, tensor, device=device)
                            fake_p = pred_res["fake_probability"]
                            frame_scores.append(fake_p)

                            # Generate explanation overlay
                            overlay = None
                            if explainer:
                                try:
                                    heatmap = explainer.generate(tensor, target_class=1)
                                    overlay = create_heatmap_overlay(frame_np, heatmap, alpha=0.5)
                                except Exception:
                                    overlay = None

                            analyzed_data.append({
                                "frame_idx": idx,
                                "frame": frame_np,
                                "overlay": overlay,
                                "fake_prob": fake_p,
                                "real_prob": 1.0 - fake_p,
                            })

                        # Aggregate video prediction
                        mean_fake_p = float(np.mean(frame_scores))
                        is_fake = mean_fake_p >= 0.5
                        verdict = "FAKE" if is_fake else "REAL"
                        confidence = mean_fake_p if is_fake else (1.0 - mean_fake_p)

                        st.markdown('<div class="section-header">Video Analysis Results</div>', unsafe_allow_html=True)

                        # Metric cards
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            metric_card("Aggregated Verdict", verdict)
                        with c2:
                            metric_card("Confidence", f"{confidence*100:.1f}", suffix="%")
                        with c3:
                            metric_card("Fake Probability", f"{mean_fake_p*100:.1f}", suffix="%")
                        with c4:
                            metric_card("Frames Analyzed", str(len(frames)))

                        # Uncertainty alert
                        if 0.45 <= mean_fake_p <= 0.55:
                            st.warning(
                                "⚠️ **HIGH UNCERTAINTY WARNING**: The aggregated probability is within the boundary "
                                "uncertainty zone (45%–55%). This video exhibits features near the decision boundary. "
                                "Manual expert inspection is strongly advised."
                            )

                        # Frame-level probability timeline chart
                        st.markdown("#### Frame-Level Prediction Trajectory Across Video")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(frame_scores) + 1)),
                            y=frame_scores,
                            mode='lines+markers',
                            name='Fake Probability',
                            line=dict(color='#e53e3e' if is_fake else '#38a169', width=3),
                            marker=dict(size=8),
                        ))
                        fig.add_hline(
                            y=0.5, line_dash="dash", line_color="#cbd5e0",
                            annotation_text="Decision Threshold (0.50)",
                            annotation_position="bottom right"
                        )
                        fig.update_layout(
                            **create_plotly_theme(),
                            xaxis_title="Sampled Frame Index",
                            yaxis_title="Deepfake Probability",
                            yaxis_range=[0, 1.05],
                            height=350,
                            margin=dict(l=40, r=40, t=30, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Frame Gallery with Explanation Maps
                        st.markdown("#### Sampled Face Crops & Model Attribution Maps")
                        st.info("ℹ️ **Model Attribution Note:** Heatmaps represent mathematical feature attribution (where the neural network focused), not forensic proof of tampering.")

                        # Show up to 6 frames in columns
                        cols_per_row = 3
                        for r_start in range(0, min(len(analyzed_data), 6), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for c_idx, frame_item in enumerate(analyzed_data[r_start:r_start+cols_per_row]):
                                with cols[c_idx]:
                                    img_to_show = frame_item["overlay"] if frame_item["overlay"] is not None else frame_item["frame"]
                                    st.image(
                                        img_to_show,
                                        caption=f"Frame #{frame_item['frame_idx']+1} | Fake Prob: {frame_item['fake_prob']*100:.1f}%",
                                        use_container_width=True,
                                    )

                except Exception as exc:
                    st.error(f"Inference execution error: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        st.info("Select or upload a video file above to begin deepfake video analysis.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Experiment Results
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Experiment Results":
    st.markdown("""
    <div class="main-header">
        <h1>Experiment Results</h1>
        <p>All metrics from actual model predictions — never fabricated</p>
    </div>
    """, unsafe_allow_html=True)

    if data['results']:
        results = data['results']

        # Summary metrics
        tabs = st.tabs(["Smoke Tests", "Baselines", "All Experiments"])

        with tabs[0]:
            st.subheader("🧪 Synthetic Smoke Tests")
            st.warning("⚠️ These results are from SYNTHETIC data and are NOT research results. "
                      "They only validate pipeline correctness.")

            for key in ['cnn_smoke_test', 'vit_smoke_test']:
                if key in results and 'test_metrics' in results[key]:
                    m = results[key]['test_metrics']
                    model_name = results[key].get('model_name', key)
                    st.markdown(f"**{model_name}**")

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        metric_card("Accuracy", f"{m.get('accuracy', 0):.3f}")
                    with c2:
                        metric_card("AUROC", f"{m.get('auroc', 0):.3f}")
                    with c3:
                        metric_card("F1", f"{m.get('f1_score', 0):.3f}")
                    with c4:
                        metric_card("Precision", f"{m.get('precision', 0):.3f}")
                    with c5:
                        metric_card("Recall", f"{m.get('recall', 0):.3f}")

                    st.markdown("---")

        with tabs[1]:
            st.subheader("📊 Within-Dataset Baselines")

            for key in ['cnn_baseline', 'vit_baseline']:
                if key in results and isinstance(results[key], dict) and 'test_metrics' in results[key]:
                    m = results[key]['test_metrics']
                    model = results[key].get('model', key)
                    st.markdown(f"**{model}**")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        metric_card("Accuracy", f"{m.get('accuracy', 0):.3f}")
                    with c2:
                        metric_card("AUROC", f"{m.get('auroc', 0):.3f}")
                    with c3:
                        metric_card("F1", f"{m.get('f1_score', 0):.3f}")
                    with c4:
                        metric_card("EER", f"{m.get('eer', 0):.3f}")
                    st.markdown("---")

        with tabs[2]:
            st.subheader("📋 All Experiments")

            if data.get('comparison'):
                df = pd.DataFrame(data['comparison'])
                st.dataframe(df, use_container_width=True, height=400)
            elif data.get('experiments'):
                df = pd.DataFrame(data['experiments'])
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("No experiment data available yet. Run experiments first.")

    else:
        st.info("No experiment results found. Run `python scripts/run_all.py` to generate results.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Cross-Dataset Matrix
# ═══════════════════════════════════════════════════════════════
elif page == "🔄 Cross-Dataset Matrix":
    st.markdown("""
    <div class="main-header">
        <h1>Cross-Dataset Evaluation Matrix</h1>
        <p>6-pair directional evaluation with NO target fine-tuning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="research-note">
        <strong>Protocol:</strong> Models trained on the source dataset are evaluated directly
        on the target dataset without any fine-tuning, threshold tuning, or hyperparameter
        adjustment on the target data. This measures true generalization ability.
    </div>
    """, unsafe_allow_html=True)

    if data.get('cross_dataset'):
        # Build matrix heatmap
        datasets = ['faceforensics', 'celebdf', 'dfdc']
        matrix = np.zeros((len(datasets), len(datasets)))

        for i, src in enumerate(datasets):
            for j, tgt in enumerate(datasets):
                if i == j:
                    matrix[i][j] = np.nan
                else:
                    key = f"{src}_to_{tgt}"
                    if key in data['cross_dataset']:
                        matrix[i][j] = data['cross_dataset'][key].get('auroc', 0)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[d.upper() for d in datasets],
            y=[d.upper() for d in datasets],
            colorscale='RdYlGn',
            zmin=0.5, zmax=1.0,
            text=np.where(np.isnan(matrix), 'N/A',
                         np.round(matrix, 3).astype(str)),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="AUROC"),
        ))
        fig.update_layout(
            title="Cross-Dataset AUROC Matrix (Train → Test)",
            xaxis_title="Test Dataset",
            yaxis_title="Train Dataset",
            **create_plotly_theme(),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cross-dataset results not yet available.")

        # Show planned matrix
        st.subheader("Planned Evaluation Pairs")
        pairs_df = pd.DataFrame([
            {"Train": "FF++", "Test": "Celeb-DF", "Status": "Pending"},
            {"Train": "FF++", "Test": "DFDC", "Status": "Pending"},
            {"Train": "Celeb-DF", "Test": "FF++", "Status": "Pending"},
            {"Train": "Celeb-DF", "Test": "DFDC", "Status": "Pending"},
            {"Train": "DFDC", "Test": "FF++", "Status": "Pending"},
            {"Train": "DFDC", "Test": "Celeb-DF", "Status": "Pending"},
        ])
        st.dataframe(pairs_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: CNN vs ViT Comparison
# ═══════════════════════════════════════════════════════════════
elif page == "🏗️ CNN vs ViT Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>CNN vs ViT Architecture Comparison</h1>
        <p>Comparing generalization capabilities across model families</p>
    </div>
    """, unsafe_allow_html=True)

    if data['results']:
        results = data['results']

        # Gather CNN and ViT results
        comparison_data = []
        for key in ['cnn_smoke_test', 'cnn_baseline', 'vit_smoke_test', 'vit_baseline']:
            if key in results and isinstance(results[key], dict) and 'test_metrics' in results[key]:
                m = results[key]['test_metrics']
                comparison_data.append({
                    'Model': results[key].get('model_name', key),
                    'Type': 'CNN' if 'cnn' in key else 'ViT',
                    'Experiment': key,
                    'Accuracy': m.get('accuracy', 0),
                    'AUROC': m.get('auroc', 0),
                    'F1': m.get('f1_score', 0),
                    'EER': m.get('eer', 0),
                    'Precision': m.get('precision', 0),
                    'Recall': m.get('recall', 0),
                })

        # Also gather from backbone ablation
        if 'backbone_ablation' in results and isinstance(results['backbone_ablation'], list):
            for r in results['backbone_ablation']:
                m = r.get('test_metrics', {})
                model_type = 'ViT' if 'vit' in r.get('backbone', '') else 'CNN'
                comparison_data.append({
                    'Model': r.get('backbone', 'unknown'),
                    'Type': model_type,
                    'Experiment': 'backbone_ablation',
                    'Accuracy': m.get('accuracy', 0),
                    'AUROC': m.get('auroc', 0),
                    'F1': m.get('f1_score', 0),
                    'EER': m.get('eer', 0),
                    'Precision': m.get('precision', 0),
                    'Recall': m.get('recall', 0),
                })

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            # Grouped bar chart
            metrics_to_plot = ['Accuracy', 'AUROC', 'F1', 'Precision', 'Recall']
            fig = go.Figure()

            colors = {'CNN': '#667eea', 'ViT': '#764ba2'}
            for model_type in df['Type'].unique():
                subset = df[df['Type'] == model_type]
                for _, row in subset.iterrows():
                    fig.add_trace(go.Bar(
                        name=f"{row['Model']} ({model_type})",
                        x=metrics_to_plot,
                        y=[row[m] for m in metrics_to_plot],
                        marker_color=colors.get(model_type, '#f093fb'),
                    ))

            fig.update_layout(
                title="Model Performance Comparison",
                barmode='group',
                **create_plotly_theme(),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.subheader("Detailed Comparison Table")
            st.dataframe(df.round(4), use_container_width=True)
        else:
            st.info("No comparison data available yet.")
    else:
        st.info("Run experiments first to see model comparisons.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Ablation Studies
# ═══════════════════════════════════════════════════════════════
elif page == "🧪 Ablation Studies":
    st.markdown("""
    <div class="main-header">
        <h1>Ablation Studies</h1>
        <p>Controlled experiments isolating individual factors</p>
    </div>
    """, unsafe_allow_html=True)

    if data['results']:
        results = data['results']

        tabs = st.tabs(["Augmentation", "Loss Function", "Backbone Capacity"])

        # Augmentation ablation
        with tabs[0]:
            st.subheader("Gap C: Augmentation Strategy Ablation")
            st.markdown("Comparing different training augmentation strategies with identical data splits and training budget.")

            if 'augmentation_ablation' in results and isinstance(results['augmentation_ablation'], list):
                aug_data = []
                for r in results['augmentation_ablation']:
                    m = r.get('test_metrics', {})
                    aug_data.append({
                        'Strategy': r.get('strategy', 'unknown'),
                        'Accuracy': m.get('accuracy', 0),
                        'AUROC': m.get('auroc', 0),
                        'F1': m.get('f1_score', 0),
                        'FPR': m.get('fpr', 0),
                        'FNR': m.get('fnr', 0),
                    })

                df_aug = pd.DataFrame(aug_data)

                fig = go.Figure()
                for metric in ['Accuracy', 'AUROC', 'F1']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df_aug['Strategy'],
                        y=df_aug[metric],
                    ))
                fig.update_layout(
                    title="Augmentation Strategy Comparison",
                    barmode='group',
                    **create_plotly_theme(),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_aug.round(4), use_container_width=True)
            else:
                st.info("Augmentation ablation data not available yet.")

        # Loss function ablation
        with tabs[1]:
            st.subheader("Gap D: Loss Function Ablation")
            st.markdown("Comparing cross-entropy, focal loss, and weighted BCE with identical data splits.")

            if 'loss_ablation' in results and isinstance(results['loss_ablation'], list):
                loss_data = []
                for r in results['loss_ablation']:
                    m = r.get('test_metrics', {})
                    loss_data.append({
                        'Loss Function': r.get('loss_function', 'unknown'),
                        'Accuracy': m.get('accuracy', 0),
                        'AUROC': m.get('auroc', 0),
                        'F1': m.get('f1_score', 0),
                        'FPR': m.get('fpr', 0),
                        'FNR': m.get('fnr', 0),
                    })

                df_loss = pd.DataFrame(loss_data)

                fig = go.Figure()
                for metric in ['Accuracy', 'AUROC', 'F1']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df_loss['Loss Function'],
                        y=df_loss[metric],
                    ))
                fig.update_layout(
                    title="Loss Function Comparison",
                    barmode='group',
                    **create_plotly_theme(),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_loss.round(4), use_container_width=True)
            else:
                st.info("Loss ablation data not available yet.")

        # Backbone capacity ablation
        with tabs[2]:
            st.subheader("Gap E: Backbone Capacity Ablation")
            st.markdown("Comparing model architectures and sizes with identical training conditions.")

            if 'backbone_ablation' in results and isinstance(results['backbone_ablation'], list):
                bb_data = []
                for r in results['backbone_ablation']:
                    m = r.get('test_metrics', {})
                    si = r.get('model_size', {})
                    bb_data.append({
                        'Backbone': r.get('backbone', 'unknown'),
                        'Parameters': si.get('parameters_formatted', 'N/A'),
                        'Size (MB)': si.get('checkpoint_size_mb', 0),
                        'Accuracy': m.get('accuracy', 0),
                        'AUROC': m.get('auroc', 0),
                        'F1': m.get('f1_score', 0),
                    })

                df_bb = pd.DataFrame(bb_data)

                fig = make_subplots(rows=1, cols=2,
                                   subplot_titles=("Performance Metrics", "Model Size"))

                for metric in ['Accuracy', 'AUROC', 'F1']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df_bb['Backbone'],
                        y=df_bb[metric],
                    ), row=1, col=1)

                fig.add_trace(go.Bar(
                    name="Size (MB)",
                    x=df_bb['Backbone'],
                    y=df_bb['Size (MB)'],
                    marker_color='#f093fb',
                ), row=1, col=2)

                fig.update_layout(
                    title="Backbone Comparison",
                    **create_plotly_theme(),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_bb.round(4), use_container_width=True)
            else:
                st.info("Backbone ablation data not available yet.")
    else:
        st.info("Run experiments first: `python scripts/run_ablations.py`")


# ═══════════════════════════════════════════════════════════════
# PAGE: Explainability
# ═══════════════════════════════════════════════════════════════
elif page == "🎯 Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>Explainability & Attribution Maps</h1>
        <p>Gap F: Grad-CAM (CNN) and Attention Rollout (ViT) under distribution shift</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ All visualizations shown here are <strong>model attribution maps</strong> indicating
        which regions the model considers important for its prediction.
        They are NOT forensic proof of manipulation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Explanation Methods

    **Grad-CAM (CNN models)**
    - Computes gradient-weighted class activation maps from the last convolutional layer
    - Highlights spatial regions most influential for the predicted class

    **Attention Rollout (ViT models)**
    - Aggregates attention weights across all transformer layers
    - Shows which image patches the model attends to for its decision

    **Faithfulness Check**
    - Perturbation-based validation: progressively masks the most important regions
    - A faithful explanation shows prediction confidence dropping when important regions are hidden
    """)

    st.subheader("Generate Explanation")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model Type", ["CNN (EfficientNet)", "ViT (Vision Transformer)"])
    with col2:
        target_class = st.selectbox("Explain Class", ["Predicted", "Real (0)", "Fake (1)"])

    if st.button("🔍 Generate Explanation on Synthetic Sample", type="primary"):
        with st.spinner("Generating explanation..."):
            try:
                import torch
                from src.data.dataset import SyntheticDataset
                from src.models.factory import create_model
                from src.utils.config import ModelConfig
                from src.explainability.explainer import explain_prediction

                # Create synthetic sample
                dataset = SyntheticDataset(num_samples=1, seed=42)
                sample = dataset[0]
                image_tensor = sample['image'].unsqueeze(0)

                # Get original image for overlay
                img_np = (image_tensor[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                # Create model
                m_type = "cnn" if "CNN" in model_type else "vit"
                if m_type == "cnn":
                    config = ModelConfig(backbone="efficientnet_b0", type="cnn")
                else:
                    config = ModelConfig(backbone="vit_base_patch16_224", type="vit")

                model = create_model(config, "cpu")

                # Generate explanation
                result = explain_prediction(
                    model, image_tensor, m_type, img_np, "cpu"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Original Image**")
                    st.image(img_np, caption=f"Label: {sample['label'].item()}", width=250)

                with col2:
                    st.markdown("**Attribution Heatmap**")
                    from matplotlib import cm
                    heatmap_colored = (cm.jet(result['heatmap'])[:, :, :3] * 255).astype(np.uint8)
                    st.image(heatmap_colored, caption="Model attribution map", width=250)

                with col3:
                    st.markdown("**Overlay**")
                    if result['overlay'] is not None:
                        st.image(result['overlay'], caption="Heatmap overlay", width=250)

                # Prediction info
                st.markdown(f"""
                **Prediction:** {result['prediction']} (confidence: {result['confidence']:.3f})

                **Faithfulness Check:**
                - Score: {result['faithfulness']['faithfulness_score']:.4f}
                - Is Faithful: {'✅ Yes' if result['faithfulness']['is_faithful'] else '❌ No'}
                """)

            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
                st.info("This feature requires PyTorch and timm. Install with: pip install torch timm")


# ═══════════════════════════════════════════════════════════════
# PAGE: Robustness Tests
# ═══════════════════════════════════════════════════════════════
elif page == "🛡️ Robustness Tests":
    st.markdown("""
    <div class="main-header">
        <h1>Robustness & Stress Tests</h1>
        <p>Evaluation-time perturbation: compression, blur, resize</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="research-note">
        Robustness tests apply perturbations <strong>at evaluation time only</strong>.
        This is separate from training augmentation and measures the model's inherent
        robustness to post-processing degradation.
    </div>
    """, unsafe_allow_html=True)

    if data.get('robustness'):
        rob_data = data['robustness']

        rob_rows = []
        for name, metrics in rob_data.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                rob_rows.append({
                    'Perturbation': name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'AUROC': metrics.get('auroc', 0),
                    'F1': metrics.get('f1_score', 0),
                    'Drop (%)': metrics.get('robustness_drop_pct', 0),
                })

        if rob_rows:
            df_rob = pd.DataFrame(rob_rows)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='AUROC',
                x=df_rob['Perturbation'],
                y=df_rob['AUROC'],
                marker_color='#667eea',
            ))
            fig.add_trace(go.Bar(
                name='Drop (%)',
                x=df_rob['Perturbation'],
                y=df_rob['Drop (%)'],
                marker_color='#e57373',
                yaxis='y2',
            ))
            fig.update_layout(
                title="Robustness Under Perturbation",
                yaxis2=dict(title="Drop (%)", overlaying='y', side='right'),
                barmode='group',
                **create_plotly_theme(),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_rob.round(4), use_container_width=True)
    else:
        st.info("No robustness test results yet.")

        st.subheader("Planned Stress Tests")
        st.markdown("""
        | Test | Parameters |
        |------|-----------|
        | JPEG Q30 | Quality = 30 |
        | JPEG Q50 | Quality = 50 |
        | Blur K5 | Kernel = 5, σ = 1.0 |
        | Blur K7 | Kernel = 7, σ = 2.0 |
        | Resize 0.5x | Scale = 0.5 |
        | Combined | JPEG Q50 + Blur K5 + Resize 0.7 |
        """)


# ═══════════════════════════════════════════════════════════════
# PAGE: Training History
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Training History":
    st.markdown("""
    <div class="main-header">
        <h1>Training History & Curves</h1>
        <p>Loss, accuracy, and learning rate over training epochs</p>
    </div>
    """, unsafe_allow_html=True)

    # Find training history files
    history_files = list(Path("artifacts/checkpoints").glob("**/training_history.json"))

    if history_files:
        selected_history = st.selectbox(
            "Select Experiment",
            history_files,
            format_func=lambda x: x.parent.name,
        )

        with open(selected_history) as f:
            history = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            if 'train_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['train_loss'], mode='lines+markers',
                    name='Train Loss', line=dict(color='#667eea'),
                ))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_loss'], mode='lines+markers',
                    name='Val Loss', line=dict(color='#764ba2'),
                ))
            fig.update_layout(
                title='Training & Validation Loss',
                xaxis_title='Epoch', yaxis_title='Loss',
                **create_plotly_theme(), height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            if 'train_acc' in history:
                fig.add_trace(go.Scatter(
                    y=history['train_acc'], mode='lines+markers',
                    name='Train Acc', line=dict(color='#667eea'),
                ))
            if 'val_acc' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_acc'], mode='lines+markers',
                    name='Val Acc', line=dict(color='#764ba2'),
                ))
            if 'val_auroc' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_auroc'], mode='lines+markers',
                    name='Val AUROC', line=dict(color='#f093fb'),
                ))
            fig.update_layout(
                title='Accuracy & AUROC',
                xaxis_title='Epoch', yaxis_title='Score',
                **create_plotly_theme(), height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Learning rate
        if 'learning_rates' in history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['learning_rates'], mode='lines',
                name='Learning Rate', line=dict(color='#81c784'),
            ))
            fig.update_layout(
                title='Learning Rate Schedule',
                xaxis_title='Epoch', yaxis_title='LR',
                **create_plotly_theme(), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training history found. Run training to generate curves.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Run Experiments
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Run Experiments":
    st.markdown("""
    <div class="main-header">
        <h1>Run Experiments</h1>
        <p>Execute training, evaluation, and ablation experiments</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ Running experiments requires PyTorch and other ML dependencies.
        Make sure to install requirements first: <code>pip install -r requirements.txt</code>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quick Experiments")

        if st.button("🧪 Run Smoke Test (CNN)", type="primary"):
            with st.spinner("Running CNN smoke test..."):
                try:
                    from src.experiments.runner import ExperimentRunner
                    runner = ExperimentRunner()
                    result = runner.run_smoke_test("efficientnet_b0")
                    st.success(f"✅ Complete! Accuracy: {result['test_metrics']['accuracy']:.4f}")
                    st.json(result['test_metrics'])
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        if st.button("🧪 Run Smoke Test (ViT)"):
            with st.spinner("Running ViT smoke test..."):
                try:
                    from src.experiments.runner import ExperimentRunner
                    runner = ExperimentRunner()
                    result = runner.run_smoke_test("vit_base")
                    st.success(f"✅ Complete! Accuracy: {result['test_metrics']['accuracy']:.4f}")
                    st.json(result['test_metrics'])
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        st.subheader("Full Pipeline")

        if st.button("🚀 Run All Experiments"):
            with st.spinner("Running complete experimental pipeline... This may take a while."):
                try:
                    from src.experiments.runner import ExperimentRunner
                    runner = ExperimentRunner()
                    results = runner.run_all_experiments()
                    st.success("✅ All experiments complete!")
                    st.balloons()

                    # Show summary
                    for key, result in results.items():
                        if isinstance(result, dict) and 'test_metrics' in result:
                            m = result['test_metrics']
                            st.markdown(f"**{key}**: Acc={m.get('accuracy', 0):.4f}, "
                                      f"AUROC={m.get('auroc', 0):.4f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.subheader("📋 Command Line")
    st.code("""
# Train a model
python scripts/train.py --model efficientnet_b0 --dataset synthetic --epochs 10

# Run all experiments
python scripts/run_all.py

# Run specific ablations
python scripts/run_ablations.py --type augmentation --epochs 5
python scripts/run_ablations.py --type loss --epochs 5
python scripts/run_ablations.py --type backbone --epochs 5
    """, language="bash")


# ═══════════════════════════════════════════════════════════════
# PAGE: Dataset & Protocol
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Dataset & Protocol":
    st.markdown("""
    <div class="main-header">
        <h1>Dataset & Experimental Protocol</h1>
        <p>Data provenance, splitting protocol, and ethical considerations</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Datasets", "Protocol", "Limitations"])

    with tabs[0]:
        st.subheader("Dataset Overview")

        datasets_info = [
            {
                "Name": "FaceForensics++",
                "Videos": "~1000 original + 4000 manipulated",
                "Methods": "Deepfakes, Face2Face, FaceSwap, NeuralTextures",
                "Access": "Academic request form",
                "License": "Academic use only",
                "Status": "📝 Requires agreement",
            },
            {
                "Name": "Celeb-DF (v2)",
                "Videos": "~6000 (590 real + 5639 fake)",
                "Methods": "Improved DeepFake synthesis",
                "Access": "GitHub / direct download",
                "License": "Research use only",
                "Status": "✅ Available",
            },
            {
                "Name": "DFDC",
                "Videos": "~100K+ (subset used)",
                "Methods": "Multiple paid actors + methods",
                "Access": "Kaggle (AWS S3)",
                "License": "Kaggle competition agreement",
                "Status": "✅ Available (subset)",
            },
            {
                "Name": "Diffusion-Generated",
                "Videos": "TBD",
                "Methods": "Diffusion-based face manipulation",
                "Access": "Custom collection",
                "License": "TBD",
                "Status": "⏳ Pending - interface implemented",
            },
        ]

        df_datasets = pd.DataFrame(datasets_info)
        st.dataframe(df_datasets, use_container_width=True)

    with tabs[1]:
        st.subheader("Experimental Protocol")

        st.markdown("""
        ### Splitting Protocol
        1. **Video-level splits**: All frames from a single video belong to the SAME split
        2. **Identity-aware**: Where metadata allows, same identity kept in same split
        3. **Ratios**: 70% train / 15% validation / 15% test
        4. **Fixed seed**: All splits use seed=42 for reproducibility

        ### Cross-Dataset Protocol
        - **NO** fine-tuning on target dataset
        - **NO** threshold tuning on target data
        - **NO** hyperparameter adjustment using target performance
        - Models are trained to convergence on source → evaluated as-is on target

        ### Metric Computation
        - Frame-level metrics computed first
        - Video-level: mean of frame scores per video → threshold at 0.5
        - AUROC computed on continuous scores (threshold-independent)
        - EER: intersection of FPR and FNR curves
        """)

    with tabs[2]:
        st.subheader("Known Limitations")

        st.markdown("""
        1. **Compute constraints**: CPU-only environment limits model size and training duration
        2. **Dataset access**: Some datasets require agreements (FF++, DFDC)
        3. **Diffusion evaluation**: Pending suitable public dataset
        4. **Temporal modeling**: Not implemented (requires video-level architectures)
        5. **Real-world deployment**: System is research-grade, not production-ready
        6. **Bias**: Models may have demographic bias from training data composition
        7. **Adversarial robustness**: Not tested against adversarial attacks
        8. **Compression**: Only standard JPEG/blur tested, not advanced codecs
        """)


# ═══════════════════════════════════════════════════════════════
# PAGE: Documentation
# ═══════════════════════════════════════════════════════════════
elif page == "📝 Documentation":
    st.markdown("""
    <div class="main-header">
        <h1>Research Documentation</h1>
        <p>Architecture, protocols, and research documentation</p>
    </div>
    """, unsafe_allow_html=True)

    # List available docs
    docs_dir = Path("docs")
    if docs_dir.exists():
        doc_files = list(docs_dir.glob("*.md"))
        if doc_files:
            selected_doc = st.selectbox(
                "Select Document",
                doc_files,
                format_func=lambda x: x.stem.replace('_', ' ').title(),
            )
            with open(selected_doc) as f:
                content = f.read()
            st.markdown(content)
        else:
            st.info("No documentation files found in docs/ directory.")
    else:
        st.info("Documentation directory not found.")

    st.markdown("---")
    st.subheader("Project Structure")
    st.code("""
robustdeep/
├── app.py                 # This Streamlit dashboard
├── requirements.txt       # Pinned dependencies
├── configs/               # YAML configuration files
│   ├── data.yaml         # Data pipeline config
│   ├── models.yaml       # Model architecture configs
│   ├── training.yaml     # Training hyperparameters
│   └── experiments.yaml  # Experiment configurations
├── src/
│   ├── data/             # Dataset, preprocessing, augmentation
│   ├── models/           # CNN, ViT, Hybrid architectures
│   ├── training/         # Training engine, loss functions
│   ├── evaluation/       # Evaluator, cross-dataset matrix
│   ├── experiments/      # Experiment tracker, runner
│   ├── explainability/   # Grad-CAM, attention rollout
│   └── robustness/       # Stress testing
├── scripts/              # CLI scripts
├── tests/                # Unit tests
├── docs/                 # Research documentation
└── artifacts/            # Checkpoints, logs, figures
    """, language="text")
