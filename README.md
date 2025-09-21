# jtext - Japanese Text Processing CLI System

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A high-precision local text extraction system for Japanese documents, images, and audio using OCR, ASR, and LLM correction. Built for macOS with Apple Silicon support.

## 🚀 Features

### Phase 1 (MVP) - Core OCR

- **High-Precision OCR**: Tesseract + LLM correction for 90%+ accuracy on Japanese text
- **Image Processing**: Advanced preprocessing with deskewing, denoising, contrast enhancement
- **Rule-based Correction**: Common OCR error fixes for Japanese text
- **Structured Output**: Text files + detailed JSON metadata

### Phase 2 (Advanced) - Multi-Modal Processing

- **Document Processing**: PDF, DOCX, PPTX, HTML text extraction with metadata
- **Audio Transcription**: Whisper-based ASR for Japanese and English audio
- **Enhanced LLM Integration**: Ollama integration for real-time correction
- **Context-Aware Correction**: Intelligent correction based on document type and context
- **Multimodal OCR**: Vision-language models combined with traditional OCR for optimal accuracy
- **Multi-Format Support**: Images, Documents, Audio with unified processing pipeline

### System Features

- **Complete Local Processing**: No external API calls, maximum privacy and security
- **Apple Silicon Optimized**: Native performance on M1/M2/M3 Macs
- **Structured Output**: Text files + detailed JSON metadata for all formats

## 📋 Requirements

- **macOS 12.0+** (Apple Silicon required)
- **Python 3.12+**
- **16GB+ RAM** (32GB recommended)
- **20GB+ free storage**

## 🛠 Installation

### 1. Install System Dependencies

```bash
# Install Tesseract OCR with Japanese language support
brew install tesseract tesseract-lang

# Install FFmpeg for audio/video processing
brew install ffmpeg

# Install Ollama for local LLM (optional, for correction features)
brew install ollama
```

### 2. Install jtext

```bash
# Clone the repository
git clone https://github.com/jtext/jtext-local.git
cd jtext-local

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install jtext in development mode
pip install -e .
```

## 🎯 Quick Start

### Basic OCR Processing

```bash
# Extract text from an image
jtext ocr document.png

# Process multiple images with LLM correction
jtext ocr --llm-correct image1.jpg image2.png

# Process with vision analysis and LLM correction
jtext ocr --vision --llm-correct --vision-model llava document.png

# Specify output directory
jtext ocr --output-dir ./results document.png
```

### Advanced Usage

```bash
# Use specific OCR language
jtext ocr --lang jpn+eng document.png

# Enable verbose logging
jtext --verbose ocr document.png

# Process with custom LLM model
jtext ocr --llm-correct --model gpt-oss document.png
```

## 📖 Command Reference

### `jtext ocr` - Image OCR Processing

Extract text from images using Tesseract OCR with optional LLM correction.

```bash
jtext ocr <images...> [OPTIONS]

Arguments:
  IMAGES...  One or more image files to process

        Options:
          --lang TEXT           OCR language (default: jpn+eng)
          --llm-correct         Enable LLM correction
          --vision              Enable vision analysis
          --model TEXT          LLM model for correction (default: gpt-oss)
          --vision-model TEXT   Vision model for image analysis (default: llava)
          --output-dir PATH     Output directory (default: ./out)
          --verbose, -v         Enable verbose logging
```

**Supported Image Formats**: JPEG, PNG, TIFF, BMP, GIF

### `jtext ingest` - Document Processing

Extract text from structured documents (PDF, DOCX, PPTX, HTML).

```bash
jtext ingest <files...> [OPTIONS]

Arguments:
  FILES...  One or more document files to process

Options:
  --fallback-ocr        Use OCR fallback for low-quality pages
  --llm-correct         Enable LLM correction
  --output-dir PATH     Output directory (default: ./out)
```

**Supported Document Formats**: PDF, DOCX, PPTX, HTML

### `jtext transcribe` - Audio/Video Transcription

Transcribe audio and video files to text using ASR.

```bash
jtext transcribe <audio_files...> [OPTIONS]

Arguments:
  AUDIO_FILES...  One or more audio/video files to process

Options:
  --model TEXT      Whisper model size (tiny, base, small, medium, large)
  --lang TEXT       Language code (default: ja)
  --llm-correct     Enable LLM correction
  --output-dir PATH Output directory (default: ./out)
  --verbose, -v     Enable verbose logging
```

**Supported Audio Formats**: MP3, WAV, M4A, FLAC, MP4, MOV

### `jtext chat` - LLM Interaction

Interact with local LLM for text processing tasks.

```bash
jtext chat [OPTIONS]

Options:
  --prompt, -p TEXT     Text prompt for LLM (required)
  --model TEXT          LLM model to use (default: gpt-oss)
  --context, -c PATH    Context file to include
```

**LLM Models**: Supports Ollama models (llama2, codellama, mistral, etc.)

## 📊 Output Format

### Text Output

Processed text is saved as UTF-8 encoded `.txt` files:

```
./out/
├── document.txt
└── document.json
```

### Metadata Output

Detailed processing metadata is saved as JSON:

```json
{
  "source": "/path/to/input/image.png",
  "type": "image",
  "timestamp": "2025-09-21T10:30:00Z",
  "processing": {
    "pipeline": ["tesseract", "llm_correction"],
    "ocr_engine": "tesseract-5.3.3",
    "llm_model": "gpt-oss",
    "confidence": {
      "ocr_raw": 0.72,
      "llm_corrected": 0.91
    }
  },
  "correction_stats": {
    "characters_changed": 15,
    "words_changed": 8,
    "correction_ratio": 0.03,
    "correction_types": ["kanji_fix", "punctuation", "layout"]
  },
  "quality_metrics": {
    "character_count": 1234,
    "word_count": 567,
    "processing_time_sec": 12.5,
    "memory_usage_mb": 156
  }
}
```

## 🔧 Configuration

### Environment Variables

Configure jtext behavior using environment variables:

```bash
# OCR Settings
export JTEXT_OCR_LANG="jpn+eng"
export JTEXT_LLM_MODEL="gpt-oss"

# Processing Limits
export JTEXT_MAX_FILE_SIZE="2048"  # MB
export JTEXT_MEMORY_LIMIT="8192"   # MB

# Logging
export JTEXT_LOG_LEVEL="INFO"
```

### Configuration File

Create a `~/.jtext/config.yaml` file for persistent settings:

```yaml
ocr:
  default_language: "jpn+eng"
  confidence_threshold: 0.6

llm:
  default_model: "gpt-oss"
  correction_enabled: false

processing:
  max_file_size_mb: 2048.0
  max_concurrent_files: 4

output:
  default_format: "txt"
  include_metadata: true
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=jtext --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 🏗 Development

### Project Structure

```
jtext/
├── cli.py                    # CLI entry point
├── core/                     # Core processing modules
│   ├── ocr_hybrid.py        # OCR + LLM correction
│   ├── ingest.py            # Document extraction
│   └── asr.py               # Audio transcription
├── preprocessing/            # Input preprocessing
│   ├── image_prep.py        # Image enhancement
│   └── audio_prep.py        # Audio preprocessing
├── correction/               # LLM correction
│   ├── ocr_corrector.py     # OCR result correction
│   └── prompts.py           # Correction prompts
├── utils/                    # Utilities
│   ├── io_utils.py          # File I/O operations
│   ├── validation.py        # Input validation
│   └── logging.py           # Structured logging
└── config/                   # Configuration
    └── settings.py           # Settings management
```

### Code Quality

```bash
# Format code
black jtext/ tests/

# Lint code
flake8 jtext/ tests/

# Type checking
mypy jtext/

# Run all quality checks
make lint
```

## 📈 Performance

### Benchmarks

| Operation           | Performance               | Accuracy               |
| ------------------- | ------------------------- | ---------------------- |
| OCR (A4 image)      | ≤ 10 seconds              | 90%+ (with correction) |
| Document processing | ≤ 5 minutes (100MB)       | 85%+                   |
| Audio transcription | 1:3 ratio (30min → 10min) | 85%+                   |

### Memory Usage

- **Basic operation**: ≤ 8GB RAM
- **Large models**: ≤ 32GB RAM
- **Memory leak tolerance**: ≤ 1%/hour

## 🛡 Security & Privacy

- **Complete Local Processing**: No external API calls
- **Automatic Cleanup**: Temporary files deleted within 30 seconds
- **Secure File Permissions**: 600 (owner read/write only)
- **No Logging of Sensitive Data**: Automatic PII filtering

## 🐛 Troubleshooting

### Common Issues

**Tesseract not found**:

```bash
brew install tesseract tesseract-lang
```

**Permission denied errors**:

```bash
chmod +x $(which jtext)
```

**Memory errors with large files**:

```bash
export JTEXT_MEMORY_LIMIT="16384"  # 16GB
```

**Low OCR accuracy**:

- Ensure image quality is good (300+ DPI)
- Use `--llm-correct` for better results
- Check image preprocessing settings

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
jtext --verbose ocr document.png
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jtext/jtext-local/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jtext/jtext-local/discussions)
- **Documentation**: [Wiki](https://github.com/jtext/jtext-local/wiki)

## 🗺 Roadmap

### Phase 1 (Current - MVP)

- ✅ Basic OCR functionality
- ✅ Image preprocessing
- ✅ Rule-based correction
- ✅ CLI framework
- ✅ Test suite

### Phase 2 (Planned)

- 🔄 LLM integration (Ollama)
- 🔄 Document processing (PDF, DOCX)
- 🔄 Audio transcription (Whisper)
- 🔄 Advanced correction

### Phase 3 (Future)

- 📋 Web UI
- 📋 Batch processing
- 📋 Real-time processing
- 📋 Multi-language support

---

**Made with ❤️ for the Japanese text processing community**
