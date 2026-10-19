"""
Generate sample test videos (Real, Deepfake face-swap, and Diffusion-generated)
for end-to-end pipeline and dashboard testing.

Usage:
    python scripts/create_sample_videos.py
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Fix Windows cp1252 encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SAMPLE_DIR = Path("data/sample_videos")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def draw_face(canvas, center_x, center_y, scale=1.0, is_fake=False, is_diffusion=False, frame_idx=0):
    """Draw a face with landmarks, skin tones, eyes, nose, mouth."""
    h, w = canvas.shape[:2]

    # Skin color (BGR)
    skin_color = (180, 205, 240) if not is_fake else (160, 195, 235)
    shadow_color = (140, 160, 200)

    # Face oval
    axes = (int(65 * scale), int(85 * scale))
    cv2.ellipse(canvas, (center_x, center_y), axes, 0, 0, 360, skin_color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (center_x, center_y), axes, 0, 0, 360, shadow_color, 2, cv2.LINE_AA)

    # Hair
    hair_color = (40, 45, 55)
    hair_top = center_y - int(70 * scale)
    cv2.ellipse(canvas, (center_x, hair_top), (int(70 * scale), int(35 * scale)), 0, 180, 360, hair_color, -1, cv2.LINE_AA)

    # Eyes
    eye_offset_x = int(24 * scale)
    eye_y = center_y - int(12 * scale)
    eye_size = int(8 * scale)

    # Left eye
    cv2.circle(canvas, (center_x - eye_offset_x, eye_y), eye_size, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (center_x - eye_offset_x, eye_y), int(eye_size * 0.5), (70, 50, 30), -1, cv2.LINE_AA)
    cv2.circle(canvas, (center_x - eye_offset_x, eye_y), int(eye_size * 0.25), (10, 10, 10), -1, cv2.LINE_AA)

    # Right eye
    cv2.circle(canvas, (center_x + eye_offset_x, eye_y), eye_size, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (center_x + eye_offset_x, eye_y), int(eye_size * 0.5), (70, 50, 30), -1, cv2.LINE_AA)
    cv2.circle(canvas, (center_x + eye_offset_x, eye_y), int(eye_size * 0.25), (10, 10, 10), -1, cv2.LINE_AA)

    # Nose
    nose_y = center_y + int(10 * scale)
    pts = np.array([
        [center_x, nose_y - int(12 * scale)],
        [center_x - int(6 * scale), nose_y + int(6 * scale)],
        [center_x + int(6 * scale), nose_y + int(6 * scale)],
    ], np.int32)
    cv2.polylines(canvas, [pts], False, (140, 160, 200), 2, cv2.LINE_AA)

    # Mouth
    mouth_y = center_y + int(36 * scale)
    smile_delta = int(3 * np.sin(frame_idx * 0.2))
    cv2.ellipse(canvas, (center_x, mouth_y), (int(16 * scale), int(6 * scale) + smile_delta), 0, 0, 180, (80, 80, 180), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (center_x, mouth_y), (int(16 * scale), int(6 * scale) + smile_delta), 0, 0, 180, (50, 50, 140), 2, cv2.LINE_AA)

    # Artifacts for deepfakes
    if is_fake:
        # Facial boundary blending artifact (color mismatch / seam ring)
        cv2.ellipse(canvas, (center_x, center_y), (int(52 * scale), int(68 * scale)), 0, 0, 360, (190, 170, 140), 3, cv2.LINE_AA)
        # Eye jitter / asymmetric blur
        blur_region = canvas[eye_y - 15:eye_y + 15, center_x - 35:center_x + 35]
        if blur_region.size > 0:
            blurred = cv2.GaussianBlur(blur_region, (5, 5), 1.5)
            canvas[eye_y - 15:eye_y + 15, center_x - 35:center_x + 35] = blurred

    if is_diffusion:
        # Characteristic diffusion smoothing / oversaturated highlight
        cv2.circle(canvas, (center_x - eye_offset_x + 2, eye_y - 2), int(2 * scale), (255, 255, 255), -1)
        # Subtle texture noise on perimeter
        noise = (np.random.randn(*canvas.shape) * 6).astype(np.int16)
        canvas[:] = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_sample_video(filename, is_fake=False, is_diffusion=False, num_frames=30, width=320, height=320):
    """Write an MP4 video."""
    filepath = SAMPLE_DIR / filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filepath), fourcc, 20.0, (width, height))

    base_x = width // 2
    base_y = height // 2

    for i in range(num_frames):
        # Background gradient
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, 0] = np.linspace(40, 70, height)[:, None]
        canvas[:, :, 1] = np.linspace(45, 80, height)[:, None]
        canvas[:, :, 2] = np.linspace(50, 90, height)[:, None]

        # Natural subtle head movement
        shift_x = int(6 * np.sin(i * 0.15))
        shift_y = int(4 * np.cos(i * 0.12))

        draw_face(
            canvas,
            center_x=base_x + shift_x,
            center_y=base_y + shift_y,
            scale=1.1,
            is_fake=is_fake,
            is_diffusion=is_diffusion,
            frame_idx=i,
        )

        out.write(canvas)

    out.release()
    print(f"[OK] Created: {filepath} ({num_frames} frames, {width}x{height})")
    return filepath


def main():
    print("=" * 60)
    print("Creating Sample Test Videos for RobustDeep Pipeline...")
    print("=" * 60)

    # 1. Real sample video
    create_sample_video("sample_real_01.mp4", is_fake=False, is_diffusion=False)

    # 2. Fake sample video (Deepfake face-swap with seam artifacts)
    create_sample_video("sample_fake_face_swap.mp4", is_fake=True, is_diffusion=False)

    # 3. Diffusion-generated face sample video
    create_sample_video("sample_fake_diffusion.mp4", is_fake=True, is_diffusion=True)

    print("\nAll sample test videos created successfully in:", SAMPLE_DIR)


if __name__ == "__main__":
    main()
