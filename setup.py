"""
RobustDeep - Explainable Deepfake Detection Under Cross-Dataset Distribution Shift
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="robustdeep",
    version="1.0.0",
    author="RobustDeep Research Team",
    description=(
        "Explainable Vision Transformer-Based Deepfake Detection "
        "Under Cross-Dataset Distribution Shift"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "robustdeep-train=scripts.train:main",
            "robustdeep-eval=scripts.evaluate:main",
            "robustdeep-infer=scripts.infer:main",
            "robustdeep-dashboard=app:main",
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
