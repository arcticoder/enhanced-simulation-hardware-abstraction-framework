"""
Setup script for Enhanced Simulation & Hardware Abstraction Framework
"""
from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-simulation-framework",
    version="1.0.0",
    author="Enhanced Simulation Team",
    author_email="enhanced-simulation@example.com",
    description="Enhanced Simulation & Hardware Abstraction Framework with 1.2×10¹⁰× amplification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arcticoder/enhanced-simulation-hardware-abstraction-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "cuda": [
            "cupy>=10.0",
            "numba[cuda]>=0.56.0",
        ],
        "visualization": [
            "plotly>=5.0",
            "dash>=2.0",
            "mayavi>=4.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-simulation=src.enhanced_simulation_framework:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "simulation",
        "physics",
        "metamaterials", 
        "quantum",
        "field-theory",
        "multi-physics",
        "enhancement",
        "hardware-abstraction",
        "digital-twin",
        "zero-budget-validation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/arcticoder/enhanced-simulation-hardware-abstraction-framework/issues",
        "Source": "https://github.com/arcticoder/enhanced-simulation-hardware-abstraction-framework",
        "Documentation": "https://enhanced-simulation-framework.readthedocs.io/",
    },
)
