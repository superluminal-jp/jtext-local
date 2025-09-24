# jtext User Guide

## Overview

jtext is a high-precision Japanese text processing system that extracts text from images and audio files using advanced OCR, ASR, and LLM correction technologies. Built with Clean Architecture principles, it provides reliable and maintainable text processing capabilities.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd jtext-local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### System Dependencies

#### OCR Support (Tesseract)

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### LLM Support (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.2:3b
ollama pull llava:7b

# Start Ollama service
ollama serve
```

#### Audio Support (Optional)

```bash
# For audio transcription
pip install faster-whisper
```

### Basic Usage

```bash
# Check system health
jtext health

# Process an image with OCR
jtext ocr image.jpg

# Process with LLM correction
jtext ocr image.jpg --llm-correct

# Process audio file
jtext transcribe audio.wav

# View processing statistics
jtext stats
```

## Commands

### OCR Processing

Extract text from images using optical character recognition:

```bash
# Basic OCR
jtext ocr document.jpg

# Specify language
jtext ocr document.jpg --language jpn+eng

# Enable LLM correction
jtext ocr document.jpg --llm-correct

# Process multiple images
jtext ocr *.jpg *.png

# Specify output format
jtext ocr document.jpg --output-format json

# Set output directory
jtext ocr document.jpg --output-dir ./results/
```

**Supported Image Formats:**

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- GIF (.gif)

### Audio Transcription

Convert speech to text using automatic speech recognition:

```bash
# Basic transcription
jtext transcribe audio.wav

# Specify language
jtext transcribe audio.wav --language jpn

# Use specific model
jtext transcribe audio.wav --model base

# Enable LLM correction
jtext transcribe audio.wav --llm-correct

# Process multiple audio files
jtext transcribe *.wav *.mp3
```

**Supported Audio Formats:**

- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)

### Document Ingestion

Process documents (currently planned feature):

```bash
# Process PDF (future implementation)
jtext ingest document.pdf

# Process with specific type
jtext ingest --type image document.jpg
```

### System Commands

#### Health Check

Verify system components and dependencies:

```bash
jtext health
```

Example output:

```
✅ System Health Check
✅ OCR Service: Available (Tesseract)
✅ LLM Service: Available (Ollama - llama3.2:3b)
❌ ASR Service: Unavailable (faster-whisper not installed)
✅ Storage: Available
✅ Logging: Configured
```

#### Statistics

View processing statistics and system metrics:

```bash
jtext stats
```

Example output:

```
📊 Processing Statistics
Total Documents: 15
Completed: 12
Failed: 3
Average Processing Time: 2.3s
Success Rate: 80%
```

## Configuration

### Command Line Options

#### OCR Options

- `--language`: OCR language (default: jpn+eng)
- `--llm-correct`: Enable LLM-based text correction
- `--output-format`: Output format (txt, json)
- `--output-dir`: Output directory
- `--verbose`: Verbose output

#### Transcription Options

- `--language`: Audio language (default: jpn)
- `--model`: ASR model (base, small, medium, large)
- `--llm-correct`: Enable LLM-based text correction
- `--output-format`: Output format (txt, json)
- `--output-dir`: Output directory

#### Global Options

- `--help`: Show help message
- `--version`: Show version information
- `--verbose`: Enable verbose logging

### Environment Variables

```bash
# Logging
export JTEXT_LOG_LEVEL=INFO
export JTEXT_LOG_FILE=logs/jtext.log

# Model configuration
export JTEXT_LLM_MODEL=llama3.2:3b
export JTEXT_OLLAMA_BASE_URL=http://localhost:11434

# Processing limits
export JTEXT_MAX_FILE_SIZE_MB=100
export JTEXT_MAX_PROCESSING_TIME_SECONDS=300
```

## Output Formats

### Text Output

Default text output includes extracted content:

```
Extracted Text:
これは日本語のテキストです。
OCRで正確に認識されました。

Processing Information:
- File: document.jpg
- Language: Japanese
- Confidence: 95%
- Processing Time: 2.3s
```

### JSON Output

Structured output with complete metadata:

```json
{
  "document_id": "doc_abc123",
  "file_path": "document.jpg",
  "document_type": "IMAGE",
  "language": "JAPANESE",
  "processing_results": [
    {
      "result_type": "OCR",
      "content": "これは日本語のテキストです。",
      "confidence": 0.95,
      "processing_time": 2.3,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "status": "COMPLETED",
  "created_at": "2024-01-15T10:28:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

## Advanced Usage

### Batch Processing

Process multiple files efficiently:

```bash
# Process all images in directory
jtext ocr images/*.jpg

# Process with specific options
jtext ocr *.png --language jpn+eng --llm-correct --output-format json

# Process audio files
jtext transcribe audio/*.wav --model base
```

### Quality Control

Monitor and optimize processing quality:

```bash
# Verbose output with detailed information
jtext ocr document.jpg --verbose

# Check processing statistics
jtext stats

# Verify system health
jtext health
```

### Custom Output Directories

Organize processing results:

```bash
# Save to specific directory
jtext ocr document.jpg --output-dir ./results/

# Create timestamped directories
mkdir results/$(date +%Y%m%d_%H%M%S)
jtext ocr *.jpg --output-dir ./results/$(date +%Y%m%d_%H%M%S)/
```

## Best Practices

### Document Preparation

1. **Image Quality**

   - Use high-resolution images (300+ DPI recommended)
   - Ensure good contrast between text and background
   - Avoid blurry, distorted, or rotated images
   - Clean images of dust, marks, or artifacts

2. **File Organization**

   - Use descriptive filenames
   - Group similar documents together
   - Keep file sizes reasonable (< 100MB)
   - Organize by processing date or content type

3. **Language Settings**
   - Use appropriate language codes
   - Combine languages when needed (jpn+eng)
   - Consider the primary language of your documents

### Processing Strategy

1. **Start Simple**

   - Begin with basic OCR for clean, simple documents
   - Add LLM correction for important or complex documents
   - Use verbose mode to understand processing results

2. **Quality vs. Speed**

   - Basic OCR: Fast processing, good for bulk operations
   - LLM Correction: Higher accuracy, slower processing
   - Monitor system resources during processing

3. **Error Handling**
   - Check system health before large batch operations
   - Monitor processing statistics
   - Review failed processing attempts

### Performance Optimization

1. **System Resources**

   - Ensure adequate RAM (8GB+ recommended)
   - Use SSD storage for better I/O performance
   - Close unnecessary applications during processing

2. **Model Selection**

   - Use appropriate models for your accuracy needs
   - Smaller models process faster but may be less accurate
   - Larger models provide better accuracy but require more resources

3. **Batch Size**
   - Process files in reasonable batches
   - Monitor memory usage during processing
   - Consider system capabilities when planning batch operations

## Troubleshooting

### Common Issues

#### OCR Service Unavailable

**Error**: "OCR service not available"

**Solutions**:

```bash
# Check Tesseract installation
tesseract --version

# Install Tesseract (macOS)
brew install tesseract tesseract-lang

# Install Tesseract (Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# Verify installation
jtext health
```

#### LLM Service Unavailable

**Error**: "LLM service unavailable" or "Connection refused"

**Solutions**:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.2:3b

# Check service status
jtext health
```

#### Audio Processing Unavailable

**Error**: "ASR service unavailable"

**Solutions**:

```bash
# Install audio processing dependencies
pip install faster-whisper

# Verify installation
python -c "import faster_whisper; print('Audio support available')"

# Check system health
jtext health
```

#### Low Processing Accuracy

**Symptoms**: Poor text extraction quality

**Solutions**:

1. Check image quality (resolution, clarity, contrast)
2. Verify correct language settings
3. Enable LLM correction for better accuracy
4. Use verbose mode to understand processing details

```bash
# Enable LLM correction
jtext ocr document.jpg --llm-correct

# Use verbose output for debugging
jtext ocr document.jpg --verbose

# Try different language combinations
jtext ocr document.jpg --language jpn+eng
```

#### Memory Issues

**Error**: "Out of memory" or slow processing

**Solutions**:

1. Process files individually instead of batches
2. Reduce file sizes if possible
3. Close unnecessary applications
4. Increase system virtual memory

```bash
# Process files one at a time
for file in *.jpg; do
    jtext ocr "$file"
done

# Monitor system resources
top  # or htop on Linux/macOS
```

### Debugging

#### Enable Debug Logging

```bash
# Set debug log level
export JTEXT_LOG_LEVEL=DEBUG

# View detailed processing information
jtext ocr document.jpg --verbose
```

#### Check Log Files

```bash
# View log file (if configured)
tail -f logs/jtext.log

# Check for error messages
grep ERROR logs/jtext.log
```

#### System Diagnostics

```bash
# Complete system health check
jtext health

# View processing statistics
jtext stats

# Test with known good file
jtext ocr path/to/simple/test/image.jpg
```

## Integration

### Python API

For programmatic access, use the Python API:

```python
from jtext import (
    ProcessDocumentRequest, ProcessDocumentUseCase,
    DocumentType, Language
)

# Create processing request
request = ProcessDocumentRequest(
    file_path="document.jpg",
    document_type=DocumentType.IMAGE,
    language=Language.JAPANESE,
    enable_correction=True
)

# Process document
result = use_case.execute(request)

if result.is_ok:
    response = result.unwrap()
    print(f"Extracted text: {response.processing_results[0]['content']}")
else:
    print(f"Error: {result.unwrap_err()}")
```

### Scripting

Automate processing with shell scripts:

```bash
#!/bin/bash
# Process all images in a directory

INPUT_DIR="./documents"
OUTPUT_DIR="./results"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.{jpg,png,tiff}; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        jtext ocr "$file" --llm-correct --output-dir "$OUTPUT_DIR"
    fi
done

echo "Processing complete. Results in $OUTPUT_DIR"
```

## Support

### Getting Help

- **Documentation**: Check this user guide and API documentation
- **System Health**: Run `jtext health` to check system status
- **Statistics**: Run `jtext stats` to view processing metrics
- **Verbose Mode**: Use `--verbose` flag for detailed information

### Reporting Issues

When reporting issues, please include:

1. **System Information**:

   - Operating system and version
   - Python version
   - jtext version (`jtext --version`)

2. **Error Information**:

   - Complete error messages
   - Log file contents (if available)
   - System health output (`jtext health`)

3. **Reproduction Steps**:
   - Exact commands used
   - Sample files (if possible)
   - Expected vs. actual behavior

### Performance Benchmarks

Typical processing times on modern hardware:

- **Simple Image OCR**: 1-3 seconds per image
- **OCR with LLM Correction**: 3-8 seconds per image
- **Audio Transcription**: 5-15 seconds per minute of audio
- **Batch Processing**: Scales linearly with file count

Memory usage typically ranges from 100-500MB per processing operation, depending on file size and enabled features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

# Install Tesseract (Ubuntu)

sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# Verify installation

jtext health

````

#### LLM Service Unavailable

**Error**: "LLM service unavailable" or "Connection refused"

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.2:3b

# Check service status
jtext health
````

#### Audio Processing Unavailable

**Error**: "ASR service unavailable"

**Solutions**:

```bash
# Install audio processing dependencies
pip install faster-whisper

# Verify installation
python -c "import faster_whisper; print('Audio support available')"

# Check system health
jtext health
```

#### Low Processing Accuracy

**Symptoms**: Poor text extraction quality

**Solutions**:

1. Check image quality (resolution, clarity, contrast)
2. Verify correct language settings
3. Enable LLM correction for better accuracy
4. Use verbose mode to understand processing details

```bash
# Enable LLM correction
jtext ocr document.jpg --llm-correct

# Use verbose output for debugging
jtext ocr document.jpg --verbose

# Try different language combinations
jtext ocr document.jpg --language jpn+eng
```

#### Memory Issues

**Error**: "Out of memory" or slow processing

**Solutions**:

1. Process files individually instead of batches
2. Reduce file sizes if possible
3. Close unnecessary applications
4. Increase system virtual memory

```bash
# Process files one at a time
for file in *.jpg; do
    jtext ocr "$file"
done

# Monitor system resources
top  # or htop on Linux/macOS
```

### Debugging

#### Enable Debug Logging

```bash
# Set debug log level
export JTEXT_LOG_LEVEL=DEBUG

# View detailed processing information
jtext ocr document.jpg --verbose
```

#### Check Log Files

```bash
# View log file (if configured)
tail -f logs/jtext.log

# Check for error messages
grep ERROR logs/jtext.log
```

#### System Diagnostics

```bash
# Complete system health check
jtext health

# View processing statistics
jtext stats

# Test with known good file
jtext ocr path/to/simple/test/image.jpg
```

## Integration

### Python API

For programmatic access, use the Python API:

```python
from jtext import (
    ProcessDocumentRequest, ProcessDocumentUseCase,
    DocumentType, Language
)

# Create processing request
request = ProcessDocumentRequest(
    file_path="document.jpg",
    document_type=DocumentType.IMAGE,
    language=Language.JAPANESE,
    enable_correction=True
)

# Process document
result = use_case.execute(request)

if result.is_ok:
    response = result.unwrap()
    print(f"Extracted text: {response.processing_results[0]['content']}")
else:
    print(f"Error: {result.unwrap_err()}")
```

### Scripting

Automate processing with shell scripts:

```bash
#!/bin/bash
# Process all images in a directory

INPUT_DIR="./documents"
OUTPUT_DIR="./results"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.{jpg,png,tiff}; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        jtext ocr "$file" --llm-correct --output-dir "$OUTPUT_DIR"
    fi
done

echo "Processing complete. Results in $OUTPUT_DIR"
```

## Support

### Getting Help

- **Documentation**: Check this user guide and API documentation
- **System Health**: Run `jtext health` to check system status
- **Statistics**: Run `jtext stats` to view processing metrics
- **Verbose Mode**: Use `--verbose` flag for detailed information

### Reporting Issues

When reporting issues, please include:

1. **System Information**:

   - Operating system and version
   - Python version
   - jtext version (`jtext --version`)

2. **Error Information**:

   - Complete error messages
   - Log file contents (if available)
   - System health output (`jtext health`)

3. **Reproduction Steps**:
   - Exact commands used
   - Sample files (if possible)
   - Expected vs. actual behavior

### Performance Benchmarks

Typical processing times on modern hardware:

- **Simple Image OCR**: 1-3 seconds per image
- **OCR with LLM Correction**: 3-8 seconds per image
- **Audio Transcription**: 5-15 seconds per minute of audio
- **Batch Processing**: Scales linearly with file count

Memory usage typically ranges from 100-500MB per processing operation, depending on file size and enabled features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Overview

jtext is a high-precision Japanese text processing system that extracts text from various document types including images, PDFs, audio files, and structured documents. The system uses advanced OCR, vision-language models, and LLM correction to achieve 95%+ accuracy.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jtext/jtext-local.git
cd jtext-local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Process a single image
jtext ocr image.png

# Process with LLM correction
jtext ocr image.png --llm-correct

# Process with vision analysis
jtext ocr image.png --vision

# Process multiple images
jtext ocr image1.png image2.jpg image3.tiff

# Process audio files
jtext transcribe audio.mp3

# Process documents
jtext ingest document.pdf
```

## Features

### 🖼️ Advanced Multimodal OCR

Process images with cutting-edge OCR technology:

```bash
# Basic OCR
jtext ocr document.png

# OCR with LLM correction
jtext ocr document.png --llm-correct --model llama3.2:3b

# OCR with vision analysis
jtext ocr document.png --vision --vision-model gemma3:4b

# Combined processing
jtext ocr document.png --llm-correct --vision
```

**Supported Image Formats:**

- PNG, JPEG, TIFF, BMP, WEBP
- High-resolution images (up to 4K)
- Scanned documents and photos

### 🎵 Audio Transcription

Transcribe Japanese and English audio:

```bash
# Basic transcription
jtext transcribe audio.mp3

# High-quality transcription
jtext transcribe audio.mp3 --model large --lang ja

# Transcription with LLM correction
jtext transcribe audio.mp3 --llm-correct
```

**Supported Audio Formats:**

- MP3, WAV, M4A, FLAC, OGG
- Video files (MP4, AVI, MOV)

### 📄 Document Processing

Extract text from structured documents:

```bash
# PDF documents
jtext ingest document.pdf

# Office documents
jtext ingest presentation.pptx
jtext ingest report.docx

# Web content
jtext ingest webpage.html

# Batch processing
jtext ingest *.pdf *.docx *.pptx
```

**Supported Document Types:**

- PDF (with OCR fallback for scanned pages)
- Microsoft Office (DOCX, PPTX)
- HTML web pages
- Plain text files

## Configuration

### Environment Variables

```bash
# Model configuration
export JTEXT_LLM_MODEL="llama3.2:3b"
export JTEXT_VISION_MODEL="gemma3:4b"
export JTEXT_ASR_MODEL="whisper-base"

# Processing limits
export JTEXT_MAX_FILE_SIZE="2048"  # MB
export JTEXT_MEMORY_LIMIT="8192"   # MB

# Logging
export JTEXT_LOG_LEVEL="INFO"
```

### Configuration File

Create `~/.jtext/config.yaml`:

```yaml
# Model settings
models:
  llm:
    default: "llama3.2:3b"
    temperature: 0.1
    max_tokens: 2048

  vision:
    default: "gemma3:4b"
    temperature: 0.1
    max_tokens: 1024

  asr:
    default: "whisper-base"
    language: "ja"

# Processing settings
processing:
  max_file_size_mb: 2048
  max_concurrent_files: 4
  timeout_seconds: 300

# Output settings
output:
  format: "txt"
  include_metadata: true
  encoding: "utf-8"
```

## Advanced Usage

### Batch Processing

Process multiple files efficiently:

```bash
# Process all images in a directory
jtext ocr images/*.png --llm-correct

# Process mixed file types
jtext ocr *.png *.jpg --llm-correct --vision

# Process with custom output directory
jtext ocr *.png -o ./results/
```

### Quality Control

Monitor processing quality:

```bash
# Verbose output with quality metrics
jtext ocr document.png -v

# Get processing statistics
jtext stats

# Check system health
jtext health
```

### Custom Models

Use custom LLM models:

```bash
# Use specific model
jtext ocr document.png --model llama3.2:7b

# Use custom vision model
jtext ocr document.png --vision --vision-model llava:13b
```

## Output Formats

### Text Output

Default text output includes:

- Extracted text
- Confidence scores
- Processing metadata
- Quality metrics

### JSON Output

Structured output with full metadata:

```json
{
  "source": "/path/to/document.png",
  "type": "image",
  "timestamp": "2024-01-15T10:30:00Z",
  "processing": {
    "pipeline": ["tesseract", "vision_analysis", "llm_correction"],
    "fusion_method": "multimodal_fusion",
    "confidence": {
      "ocr_raw": 0.85,
      "vision_analysis": 0.9,
      "llm_corrected": 0.92
    }
  },
  "extracted_text": "処理されたテキスト",
  "quality_metrics": {
    "character_count": 150,
    "word_count": 25,
    "processing_time_sec": 15.5,
    "memory_usage_mb": 256.0
  }
}
```

## Troubleshooting

### Common Issues

**1. Low OCR Accuracy**

```bash
# Try with preprocessing
jtext ocr document.png --preprocess

# Use vision analysis
jtext ocr document.png --vision

# Apply LLM correction
jtext ocr document.png --llm-correct
```

**2. Slow Processing**

```bash
# Use smaller models
jtext ocr document.png --model llama3.2:3b

# Disable vision analysis for simple documents
jtext ocr document.png --no-vision

# Process in smaller batches
jtext ocr *.png --batch-size 2
```

**3. Memory Issues**

```bash
# Reduce memory usage
export JTEXT_MEMORY_LIMIT="4096"

# Use smaller models
jtext ocr document.png --model llama3.2:3b

# Process files individually
jtext ocr document1.png
jtext ocr document2.png
```

### Performance Optimization

**For High-Volume Processing:**

```bash
# Use batch processing
jtext ocr *.png --batch-size 4

# Disable unnecessary features
jtext ocr *.png --no-vision --no-correction

# Use faster models
jtext ocr *.png --model llama3.2:3b
```

**For Maximum Accuracy:**

```bash
# Enable all features
jtext ocr document.png --llm-correct --vision

# Use larger models
jtext ocr document.png --model llama3.2:7b --vision-model llava:13b

# Process with high-quality settings
jtext ocr document.png --quality high
```

## Best Practices

### Document Preparation

1. **Image Quality**

   - Use high-resolution images (300+ DPI)
   - Ensure good contrast and lighting
   - Avoid blurry or distorted images

2. **File Organization**

   - Use descriptive filenames
   - Group similar documents together
   - Keep file sizes reasonable (< 2GB)

3. **Processing Strategy**
   - Start with basic OCR for simple documents
   - Use vision analysis for complex layouts
   - Apply LLM correction for important documents

### Performance Tips

1. **Batch Processing**

   - Process multiple files together
   - Use appropriate batch sizes
   - Monitor system resources

2. **Model Selection**

   - Use smaller models for speed
   - Use larger models for accuracy
   - Balance performance and quality

3. **Resource Management**
   - Monitor memory usage
   - Use SSD storage for better I/O
   - Close unnecessary applications

## Support

### Getting Help

- **Documentation**: [GitHub Wiki](https://github.com/jtext/jtext-local/wiki)
- **Issues**: [GitHub Issues](https://github.com/jtext/jtext-local/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jtext/jtext-local/discussions)

### Reporting Issues

When reporting issues, please include:

- jtext version
- Operating system
- Error messages and logs
- Sample files (if possible)
- Steps to reproduce

### Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
