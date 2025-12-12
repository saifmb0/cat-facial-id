"""Setup configuration for Cat Facial ID System."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cat-facial-id",
    version="1.0.0",
    author="Saif M.",
    description="Production-ready cat facial identification system using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saifmb0/cat-facial-id",
    project_urls={
        "Bug Tracker": "https://github.com/saifmb0/cat-facial-id/issues",
        "Documentation": "https://github.com/saifmb0/cat-facial-id/docs",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "faiss-cpu>=1.7.4",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.1",
        ],
    },
)
