"""
Setup script for jtext - Japanese Text Processing CLI System.

This script configures the package installation and entry points
for the jtext command-line interface.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="jtext",
    version="1.0.0",
    author="jtext Development Team",
    author_email="dev@jtext.local",
    description="High-precision local text extraction for Japanese documents, images, and audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtext/jtext-local",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.6.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "jtext=jtext.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ocr, asr, japanese, text-processing, tesseract, whisper, llm",
    project_urls={
        "Bug Reports": "https://github.com/jtext/jtext-local/issues",
        "Source": "https://github.com/jtext/jtext-local",
        "Documentation": "https://github.com/jtext/jtext-local/wiki",
    },
)
