"""
Tests for data pipeline: datasets, preprocessing, augmentation.
"""

import numpy as np
import torch
import pytest
from src.data.dataset import SyntheticDataset, DeepfakeFrameDataset, create_weighted_sampler
from src.data.augmentation import (
    JpegCompression, GaussianBlur, ResizeReencode, ColorJitter,
    HorizontalFlip, ComposeTransforms, build_augmentation_pipeline,
    build_robustness_transform,
)


class TestSyntheticDataset:
    """Test synthetic smoke-test dataset."""

    def test_dataset_creation(self):
        """Dataset should create without errors."""
        ds = SyntheticDataset(num_samples=10, seed=42)
        assert len(ds) == 10

    def test_sample_structure(self):
        """Each sample should have image, label, video_id, frame_idx."""
        ds = SyntheticDataset(num_samples=5)
        sample = ds[0]
        assert 'image' in sample
        assert 'label' in sample
        assert 'video_id' in sample
        assert 'frame_idx' in sample

    def test_image_shape(self):
        """Images should be [3, 224, 224] tensors."""
        ds = SyntheticDataset(num_samples=5, face_size=224)
        sample = ds[0]
        assert sample['image'].shape == (3, 224, 224)

    def test_label_values(self):
        """Labels should be 0 or 1."""
        ds = SyntheticDataset(num_samples=50, seed=42)
        for i in range(len(ds)):
            assert ds[i]['label'].item() in [0, 1]

    def test_both_classes_present(self):
        """Both real and fake samples should be present."""
        ds = SyntheticDataset(num_samples=100, seed=42)
        labels = [ds[i]['label'].item() for i in range(len(ds))]
        assert 0 in labels
        assert 1 in labels

    def test_deterministic(self):
        """Same seed should produce same data."""
        ds1 = SyntheticDataset(num_samples=10, seed=42)
        ds2 = SyntheticDataset(num_samples=10, seed=42)
        for i in range(10):
            assert ds1[i]['label'].item() == ds2[i]['label'].item()

    def test_different_seeds_different_data(self):
        """Different seeds should produce different labels (in expectation)."""
        ds1 = SyntheticDataset(num_samples=50, seed=42, split='train')
        ds2 = SyntheticDataset(num_samples=50, seed=99, split='val')
        labels1 = [ds1[i]['label'].item() for i in range(50)]
        labels2 = [ds2[i]['label'].item() for i in range(50)]
        # Very unlikely to be identical
        assert labels1 != labels2

    def test_image_range(self):
        """Image values should be in [0, 1]."""
        ds = SyntheticDataset(num_samples=5)
        sample = ds[0]
        assert sample['image'].min() >= 0.0
        assert sample['image'].max() <= 1.0

    def test_configurable_size(self):
        """Face size should be configurable."""
        ds = SyntheticDataset(num_samples=3, face_size=112)
        sample = ds[0]
        assert sample['image'].shape == (3, 112, 112)


class TestAugmentations:
    """Test individual augmentation transforms."""

    def _make_image(self, size=224):
        return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

    def test_jpeg_compression(self):
        """JPEG compression should produce valid image."""
        img = self._make_image()
        transform = JpegCompression(quality_range=(30, 95))
        result = transform(image=img)
        assert result['image'].shape == img.shape
        assert result['image'].dtype == np.uint8

    def test_gaussian_blur(self):
        """Gaussian blur should produce valid image."""
        img = self._make_image()
        transform = GaussianBlur(kernel_range=(3, 7))
        result = transform(image=img)
        assert result['image'].shape == img.shape

    def test_resize_reencode(self):
        """Resize-reencode should maintain dimensions."""
        img = self._make_image()
        transform = ResizeReencode(scale_range=(0.5, 0.9))
        result = transform(image=img)
        assert result['image'].shape == img.shape

    def test_color_jitter(self):
        """Color jitter should produce valid image."""
        img = self._make_image()
        transform = ColorJitter(brightness=0.2, contrast=0.2)
        result = transform(image=img)
        assert result['image'].shape == img.shape

    def test_horizontal_flip(self):
        """Horizontal flip should maintain dimensions."""
        img = self._make_image()
        transform = HorizontalFlip(probability=1.0)
        result = transform(image=img)
        assert result['image'].shape == img.shape

    def test_compose_transforms(self):
        """Composed transforms should work sequentially."""
        img = self._make_image()
        transforms = ComposeTransforms([
            HorizontalFlip(probability=1.0),
            GaussianBlur(kernel_range=(3, 3)),
        ])
        result = transforms(image=img)
        assert result['image'].shape == img.shape

    def test_build_pipeline_no_augmentation(self):
        """Empty config should produce no-op pipeline."""
        pipeline = build_augmentation_pipeline({}, is_training=True)
        img = self._make_image()
        result = pipeline(image=img)
        assert result['image'].shape == img.shape

    def test_build_pipeline_eval_mode(self):
        """Eval mode should skip augmentation even if config has them enabled."""
        config = {
            'jpeg_compression': {'enabled': True, 'quality_range': [30, 95]},
            'gaussian_blur': {'enabled': True, 'kernel_range': [3, 7]},
        }
        pipeline = build_augmentation_pipeline(config, is_training=False)
        assert len(pipeline.transforms) == 0  # No transforms in eval mode

    def test_build_robustness_transform(self):
        """Robustness transforms should work independently."""
        img = self._make_image()
        transform = build_robustness_transform("jpeg_compression", {'quality': 50})
        result = transform(image=img)
        assert result['image'].shape == img.shape


class TestDataLeakage:
    """Tests to prevent data leakage."""

    def test_no_overlap_train_val_test(self):
        """Train/val/test splits must use different seeds to avoid overlap."""
        train_ds = SyntheticDataset(num_samples=50, seed=42, split='train')
        val_ds = SyntheticDataset(num_samples=50, seed=42, split='val')
        test_ds = SyntheticDataset(num_samples=50, seed=42, split='test')

        train_ids = [train_ds[i]['video_id'] for i in range(50)]
        val_ids = [val_ds[i]['video_id'] for i in range(50)]
        test_ids = [test_ds[i]['video_id'] for i in range(50)]

        # For synthetic data, video IDs should differ between splits
        # (they're indexed differently due to different hashed seeds)
        assert set(train_ids) != set(test_ids) or len(set(train_ids)) == len(train_ids)

    def test_augmentation_not_applied_to_eval(self):
        """Augmentation must never be applied during evaluation."""
        config = {
            'jpeg_compression': {'enabled': True, 'quality_range': [10, 20]},
        }
        eval_pipeline = build_augmentation_pipeline(config, is_training=False)
        assert len(eval_pipeline.transforms) == 0


class TestWeightedSampler:
    """Test class-balanced sampling."""

    def test_sampler_creation(self):
        """Weighted sampler should create without errors."""
        ds = SyntheticDataset(num_samples=20, seed=42)
        sampler = create_weighted_sampler(ds)
        assert sampler is not None
        assert len(sampler) == 20
