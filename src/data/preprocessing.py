"""
Video preprocessing pipeline: frame extraction, face detection, alignment, cropping.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from src.utils.logging_utils import get_logger

logger = get_logger("preprocessing")


class VideoPreprocessor:
    """
    Complete video preprocessing pipeline.
    Video → Frame Sampling → Face Detection → Alignment → Normalized Crops
    """

    def __init__(
        self,
        face_size: int = 224,
        frames_per_video: int = 16,
        sampling_strategy: str = "uniform",
        face_confidence_threshold: float = 0.9,
        face_margin: float = 0.3,
        cache_dir: Optional[str] = None,
    ):
        self.face_size = face_size
        self.frames_per_video = frames_per_video
        self.sampling_strategy = sampling_strategy
        self.face_confidence_threshold = face_confidence_threshold
        self.face_margin = face_margin
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._face_detector = None

    def _get_face_detector(self):
        """Lazy-load face detector."""
        if self._face_detector is None:
            try:
                from facenet_pytorch import MTCNN
                self._face_detector = MTCNN(
                    keep_all=False,
                    post_process=False,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7],
                    device='cpu'
                )
                logger.info("Using MTCNN face detector")
            except ImportError:
                logger.warning("facenet_pytorch not available. Using OpenCV Haar cascade.")
                self._face_detector = "haar"
        return self._face_detector

    def extract_frames(
        self, video_path: str, num_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Extract frames from a video using the configured sampling strategy."""
        if num_frames is None:
            num_frames = self.frames_per_video

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error(f"Video has no frames: {video_path}")
            cap.release()
            return []

        # Determine frame indices
        if self.sampling_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif self.sampling_strategy == "random":
            indices = sorted(np.random.choice(total_frames, min(num_frames, total_frames), replace=False))
        else:  # first N frames
            indices = list(range(min(num_frames, total_frames)))

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop the largest face from a frame."""
        detector = self._get_face_detector()

        if detector == "haar":
            return self._detect_face_haar(frame)

        try:
            from facenet_pytorch import MTCNN
            import torch

            pil_img = Image.fromarray(frame)
            boxes, probs = detector.detect(pil_img)

            if boxes is None or len(boxes) == 0:
                return self._center_crop_fallback(frame)

            # Get largest face
            best_idx = 0
            if probs is not None:
                best_idx = np.argmax(probs)

            box = boxes[best_idx]
            x1, y1, x2, y2 = [int(b) for b in box]

            # Add margin
            w, h = x2 - x1, y2 - y1
            margin_x = int(w * self.face_margin)
            margin_y = int(h * self.face_margin)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(frame.shape[1], x2 + margin_x)
            y2 = min(frame.shape[0], y2 + margin_y)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return self._center_crop_fallback(frame)

            face = cv2.resize(face, (self.face_size, self.face_size))
            return face

        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
            return self._center_crop_fallback(frame)

    def _detect_face_haar(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Fallback face detection using OpenCV Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return self._center_crop_fallback(frame)

        # Get largest face
        areas = [w * h for (x, y, w, h) in faces]
        best_idx = np.argmax(areas)
        x, y, w, h = faces[best_idx]

        # Add margin
        margin_x = int(w * self.face_margin)
        margin_y = int(h * self.face_margin)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return self._center_crop_fallback(frame)

        face = cv2.resize(face, (self.face_size, self.face_size))
        return face

    def _center_crop_fallback(self, frame: np.ndarray) -> np.ndarray:
        """Center crop as fallback when face detection fails."""
        h, w = frame.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        crop = frame[y:y+size, x:x+size]
        return cv2.resize(crop, (self.face_size, self.face_size))

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        label: int = 0,
        video_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Process a single video: extract frames, detect faces, save crops.

        Returns list of sample dicts with keys: path, label, video_id, frame_idx
        """
        if video_id is None:
            video_id = Path(video_path).stem

        # Check cache
        if self.cache_dir and output_dir is None:
            output_dir = str(self.cache_dir / video_id)

        samples = []
        frames = self.extract_frames(video_path)

        if len(frames) == 0:
            logger.warning(f"No frames extracted from: {video_path}")
            return []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            face = self.detect_face(frame)
            if face is None:
                continue

            if output_dir:
                face_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(face_path, face_bgr)
            else:
                face_path = f"memory://{video_id}/frame_{i:04d}"

            samples.append({
                'path': face_path,
                'label': label,
                'video_id': video_id,
                'frame_idx': i,
            })

        logger.info(f"Processed {video_id}: {len(samples)}/{len(frames)} faces extracted")
        return samples

    def validate_video(self, video_path: str) -> Dict:
        """Validate a video file for integrity."""
        result = {
            'path': video_path,
            'valid': False,
            'error': None,
            'frame_count': 0,
            'fps': 0,
            'duration_s': 0,
            'resolution': (0, 0),
        }

        if not os.path.exists(video_path):
            result['error'] = "File not found"
            return result

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['error'] = "Cannot open video file"
                return result

            result['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result['fps'] = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result['resolution'] = (w, h)

            if result['fps'] > 0 and result['frame_count'] > 0:
                result['duration_s'] = result['frame_count'] / result['fps']

            # Try reading first frame
            ret, _ = cap.read()
            if ret:
                result['valid'] = True
            else:
                result['error'] = "Cannot read first frame"

            cap.release()
        except Exception as e:
            result['error'] = str(e)

        return result
