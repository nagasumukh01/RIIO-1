"""
RobustDeep — Explainable ViT-Based Video Deepfake Detection Under Cross-Dataset Distribution Shift
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read base requirements (skip comments, blank lines, and conditional markers)
def parse_requirements(path):
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip blanks, comments, and lines with platform conditionals (handled by pip)
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


setup(
    name="robustdeep",
    version="1.0.0",
    author="RobustDeep Research Team",
    description=(
        "Explainable Vision Transformer-Based Video Deepfake Detection "
        "Under Cross-Dataset Distribution Shift"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    # Discover both top-level src package and all sub-packages
    packages=find_packages(exclude=["tests*", "scripts*", "docs*", "configs*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "albumentations>=1.3.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "explainability": [
            "pytorch-grad-cam>=1.4.8",
        ],
        "data": [
            "gdown>=4.7.0",
            "facenet-pytorch>=2.5.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "robustdeep-train=scripts.train:main",
            "robustdeep-eval=scripts.evaluate:main",
            "robustdeep-infer=scripts.infer:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
