"""
setup.py — Install the project as a package for clean imports.
Run: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="chest_xray_detection",
    version="2.0.0",
    description="Image-Based Disease Detection from Chest X-Rays",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=9.5.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
    ],
)
