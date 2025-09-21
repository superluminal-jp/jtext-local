#!/bin/bash

# Installation script for jtext - Japanese Text Processing CLI System
# This script sets up the development environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Installing jtext - Japanese Text Processing CLI System"
echo "=================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This system is designed for macOS with Apple Silicon"
    echo "   Current OS: $OSTYPE"
    exit 1
fi

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.12"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.12+ is required"
    echo "   Current version: $python_version"
    echo "   Please install Python 3.12 or later"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if Homebrew is installed
echo "🔍 Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "❌ Error: Homebrew is not installed"
    echo "   Please install Homebrew first: https://brew.sh/"
    exit 1
fi

echo "✅ Homebrew found"

# Install system dependencies
echo "📦 Installing system dependencies..."

echo "  Installing Tesseract OCR..."
if ! brew list tesseract &> /dev/null; then
    brew install tesseract
else
    echo "    ✅ Tesseract already installed"
fi

echo "  Installing Tesseract language packs..."
if ! brew list tesseract-lang &> /dev/null; then
    brew install tesseract-lang
else
    echo "    ✅ Tesseract language packs already installed"
fi

echo "  Installing FFmpeg..."
if ! brew list ffmpeg &> /dev/null; then
    brew install ffmpeg
else
    echo "    ✅ FFmpeg already installed"
fi

echo "  Installing Ollama (optional, for LLM features)..."
if ! brew list ollama &> /dev/null; then
    brew install ollama
    echo "    ℹ️  Ollama installed. Run 'ollama serve' to start the LLM server"
else
    echo "    ✅ Ollama already installed"
fi

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install jtext in development mode
echo "🔧 Installing jtext in development mode..."
pip install -e .

# Run basic tests
echo "🧪 Running basic tests..."
if python -m pytest tests/test_validation.py -v; then
    echo "✅ Basic tests passed"
else
    echo "⚠️  Some tests failed, but installation completed"
fi

# Check installation
echo "🔍 Verifying installation..."
if jtext --version &> /dev/null; then
    echo "✅ jtext CLI installed successfully"
    jtext --version
else
    echo "❌ jtext CLI installation failed"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Test OCR functionality: jtext ocr --help"
echo "   3. Process an image: jtext ocr your_image.png"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "🔧 Development commands:"
echo "   make test        - Run all tests"
echo "   make lint        - Run linting checks"
echo "   make format      - Format code"
echo "   make clean       - Clean build artifacts"
echo ""
