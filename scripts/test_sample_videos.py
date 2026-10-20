"""
Test frame extraction and inference on sample videos.
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.infer import extract_video_frames, load_model, predict_image
import numpy as np
import torch

videos = list(Path("data/sample_videos").glob("*.mp4"))
print(f"Testing {len(videos)} sample videos...")

model, config = load_model("efficientnet_b0", checkpoint_path=None, device="cpu")

for v in videos:
    print(f"\nProcessing {v.name} (size: {v.stat().st_size} bytes)...")
    frames = extract_video_frames(str(v), num_frames=5, face_size=224)
    print(f"  Extracted {len(frames)} frames.")
    scores = []
    for f in frames:
        norm = (f.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
        res = predict_image(model, tensor, device="cpu")
        scores.append(res["fake_probability"])
    avg_score = np.mean(scores)
    verdict = "FAKE" if avg_score >= 0.5 else "REAL"
    print(f"  Verdict: {verdict} (score: {avg_score:.4f}, frames: {len(scores)})")

print("\nAll sample videos tested successfully!")
