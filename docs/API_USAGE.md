# API Usage Guide

This guide demonstrates how to use the jtext Python API for programmatic document processing.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Document Processing](#document-processing)
- [Service Configuration](#service-configuration)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)
- [Performance Optimization](#performance-optimization)

## Quick Start

### Basic Document Processing

```python
from jtext import (
    ProcessDocumentRequest, ProcessDocumentUseCase,
    DocumentType, Language,
    InMemoryDocumentRepository, InMemoryProcessingResultRepository,
    EventPublisherService, TesseractOCRService
)

# Initialize dependencies
document_repo = InMemoryDocumentRepository()
result_repo = InMemoryProcessingResultRepository()
event_publisher = EventPublisherService()

# Create use case
use_case = ProcessDocumentUseCase(
    document_repository=document_repo,
    processing_result_repository=result_repo,
    event_publisher=event_publisher
)

# Process an image
request = ProcessDocumentRequest(
    file_path="path/to/image.jpg",
    document_type=DocumentType.IMAGE,
    language=Language.JAPANESE,
    enable_correction=True
)

result = use_case.execute(request)

if result.is_ok:
    response = result.unwrap()
    print(f"Document ID: {response.document_id}")
    print(f"Processing time: {response.processing_time}s")
    for result_data in response.processing_results:
        print(f"Content: {result_data['content']}")
        print(f"Confidence: {result_data['confidence']}")
else:
    print(f"Error: {result.unwrap_err()}")
```

### CLI Usage

```bash
# Process image with OCR
jtext ocr image.jpg --language jpn+eng --llm-correct

# Transcribe audio
jtext transcribe audio.wav --model base --language jpn

# Check system health
jtext health

# View processing statistics
jtext stats
```

## Core Concepts

### Result Type

The API uses Railway-Oriented Programming with `Result` types:

```python
from jtext.core import Result, Ok, Err

# Function returns Result[T, E]
def process_file(path: str) -> Result[str, str]:
    if not os.path.exists(path):
        return Err("File not found")
    return Ok("File processed successfully")

# Handle the result
result = process_file("document.pdf")
if result.is_ok:
    content = result.unwrap()
    print(f"Success: {content}")
else:
    error = result.unwrap_err()
    print(f"Error: {error}")

# Chain operations
result = (
    validate_file(path)
    .and_then(lambda p: read_file(p))
    .and_then(lambda content: process_content(content))
    .map(lambda result: format_output(result))
)
```

### Domain Objects

#### Value Objects

```python
from jtext.domain import DocumentId, Confidence, FilePath, Language

# Document ID
doc_id = DocumentId.generate()  # Generates unique ID
doc_id = DocumentId("custom-id-123")  # Custom ID

# Confidence
confidence = Confidence.from_float(0.95)  # 95% confidence
confidence = Confidence.from_percentage(85)  # 85% confidence
print(confidence.as_percentage())  # 95.0

# File Path (validates existence)
file_path = FilePath("path/to/file.jpg")  # Validates file exists

# Language
language = Language.JAPANESE
language = Language.ENGLISH
language = Language("jpn+eng")  # Multi-language
```

#### Entities

```python
from jtext.domain import Document, ImageDocument, ProcessingResult
from datetime import datetime

# Create document
document = Document(
    id=DocumentId.generate(),
    file_path=FilePath("image.jpg"),
    document_type=DocumentType.IMAGE,
    language=Language.JAPANESE,
    status=ProcessingStatus.PENDING,
    created_at=datetime.now(),
    updated_at=datetime.now()
)

# Update document status
document.update_status(ProcessingStatus.COMPLETED)

# Create processing result
result = ProcessingResult(
    id="result-123",
    document_id=document.id,
    result_type="OCR",
    content="extracted text",
    confidence=Confidence.from_float(0.95),
    processing_time=2.5,
    created_at=datetime.now()
)
```

## Document Processing

### Image Processing (OCR)

```python
from jtext import (
    ProcessDocumentRequest, ProcessDocumentUseCase,
    DocumentType, Language, TesseractOCRService
)

# Initialize OCR service
try:
    ocr_service = TesseractOCRService()
except ImportError as e:
    print(f"OCR service not available: {e}")
    # Handle missing dependencies

# Create processing request
request = ProcessDocumentRequest(
    file_path="japanese_document.png",
    document_type=DocumentType.IMAGE,
    language=Language.JAPANESE,
    enable_correction=True,
    output_format="json"
)

# Process with OCR
result = use_case.execute(request)

if result.is_ok:
    response = result.unwrap()
    for ocr_result in response.processing_results:
        print(f"Extracted text: {ocr_result['content']}")
        print(f"Confidence: {ocr_result['confidence']}%")
        print(f"Processing time: {ocr_result['processing_time']}s")
```

### Audio Processing (ASR)

```python
from jtext.infrastructure import WhisperTranscriptionService

# Initialize transcription service
try:
    transcription_service = WhisperTranscriptionService(model_name="base")
except ImportError as e:
    print(f"Transcription service not available: {e}")

# Process audio file
request = ProcessDocumentRequest(
    file_path="japanese_audio.wav",
    document_type=DocumentType.AUDIO,
    language=Language.JAPANESE,
    enable_correction=False
)

result = use_case.execute(request)

if result.is_ok:
    response = result.unwrap()
    for transcription in response.processing_results:
        print(f"Transcribed text: {transcription['content']}")
        print(f"Confidence: {transcription['confidence']}%")
```

### LLM Text Correction

```python
from jtext.infrastructure import OllamaCorrectionService

# Initialize correction service
correction_service = OllamaCorrectionService(
    model_name="llama3.2:3b",
    base_url="http://localhost:11434"
)

# Correct text
text = "これは誤字があるテクストです。"
result = correction_service.correct_text(text, Language.JAPANESE)

if result.is_ok:
    corrected_text = result.unwrap()
    print(f"Original: {text}")
    print(f"Corrected: {corrected_text}")
else:
    print(f"Correction failed: {result.unwrap_err()}")
```

## Service Configuration

### Repository Configuration

```python
from jtext.infrastructure import (
    InMemoryDocumentRepository,
    InMemoryProcessingResultRepository
)

# In-memory repositories (current implementation)
document_repo = InMemoryDocumentRepository()
result_repo = InMemoryProcessingResultRepository()

# Future: Database repositories
# from jtext.infrastructure import PostgreSQLDocumentRepository
# document_repo = PostgreSQLDocumentRepository(connection_string)
```

### Event Publishing

```python
from jtext.infrastructure import EventPublisherService
from jtext.domain import DocumentProcessedEvent

# Initialize event publisher
event_publisher = EventPublisherService()

# Subscribe to events
def handle_document_processed(event: DocumentProcessedEvent):
    print(f"Document {event.document_id} processed successfully")

event_publisher.subscribe(handle_document_processed)

# Events are automatically published by use cases
```

### Logging Configuration

```python
from jtext.infrastructure import get_logger, set_correlation_id

# Get structured logger
logger = get_logger("MyService")

# Set correlation ID for request tracing
set_correlation_id("req-12345")

# Log with context
logger.info("Processing started", extra={
    "document_id": "doc-123",
    "file_path": "image.jpg"
})

# Logs are automatically structured with correlation IDs
```

## Error Handling

### Service Availability

```python
from jtext.infrastructure import (
    TesseractOCRService,
    WhisperTranscriptionService,
    OllamaCorrectionService
)

# Check service availability
services = {}

try:
    services['ocr'] = TesseractOCRService()
    print("✅ OCR service available")
except ImportError as e:
    print(f"❌ OCR service unavailable: {e}")

try:
    services['asr'] = WhisperTranscriptionService()
    print("✅ ASR service available")
except ImportError as e:
    print(f"❌ ASR service unavailable: {e}")

try:
    services['llm'] = OllamaCorrectionService()
    print("✅ LLM service available")
except Exception as e:
    print(f"❌ LLM service unavailable: {e}")
```

### Circuit Breaker Pattern

```python
from jtext.infrastructure import CircuitBreaker

# Create circuit breaker for external service
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=Exception
)

# Use with external service calls
@circuit_breaker
def call_external_service():
    # External service call that might fail
    return external_api.process()

# Circuit breaker automatically handles failures
try:
    result = call_external_service()
    print(f"Success: {result}")
except Exception as e:
    print(f"Service unavailable: {e}")
```

### Error Recovery

```python
from jtext.core import Result

def process_with_fallback(document_path: str) -> Result[str, str]:
    # Try primary processing
    primary_result = primary_service.process(document_path)
    if primary_result.is_ok:
        return primary_result

    # Try fallback processing
    logger.warning("Primary service failed, trying fallback")
    fallback_result = fallback_service.process(document_path)
    if fallback_result.is_ok:
        return fallback_result

    # Both failed
    return Err(f"All services failed: {primary_result.unwrap_err()}")
```

## Advanced Usage

### Batch Processing

```python
from jtext import ListDocumentsUseCase, ProcessDocumentUseCase
from pathlib import Path

def process_directory(directory_path: str) -> None:
    """Process all supported files in a directory."""

    # Get all image files
    image_files = list(Path(directory_path).glob("*.jpg")) + \
                 list(Path(directory_path).glob("*.png")) + \
                 list(Path(directory_path).glob("*.pdf"))

    # Process each file
    for file_path in image_files:
        request = ProcessDocumentRequest(
            file_path=str(file_path),
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            enable_correction=True
        )

        result = use_case.execute(request)

        if result.is_ok:
            response = result.unwrap()
            print(f"✅ Processed: {file_path.name}")
            print(f"   Document ID: {response.document_id}")
        else:
            print(f"❌ Failed: {file_path.name}")
            print(f"   Error: {result.unwrap_err()}")

# Usage
process_directory("./documents")
```

### Custom Configuration

```python
from jtext.core import AppConfig

# Create custom configuration
config = AppConfig(
    max_file_size_mb=50.0,
    max_processing_time_seconds=120,
    max_retry_attempts=3,
    default_ocr_language="jpn+eng",
    default_asr_model="base",
    default_llm_model="llama3.2:3b",
    output_directory="./custom_output"
)

# Use configuration
if config.is_file_size_valid(file_size_mb):
    # Process file
    pass
else:
    print(f"File too large: {file_size_mb}MB > {config.max_file_size_mb}MB")
```

### Performance Monitoring

```python
from jtext.core import PerformanceMonitor

# Monitor specific operations
with PerformanceMonitor("document_processing") as monitor:
    # Processing logic
    result = use_case.execute(request)

    # Add custom metrics
    monitor.add_metric("files_processed", 1)
    monitor.add_metric("total_confidence", 0.95)

# Automatic logging of performance metrics
# Metrics include: start_time, end_time, duration, memory_usage, etc.
```

### Event Handling

```python
from jtext.domain import DocumentProcessedEvent, ProcessingFailedEvent

# Custom event handlers
class ProcessingEventHandler:
    def __init__(self):
        self.processed_count = 0
        self.failed_count = 0

    def handle_processed(self, event: DocumentProcessedEvent):
        self.processed_count += 1
        print(f"Document processed: {event.document_id}")
        print(f"Total processed: {self.processed_count}")

    def handle_failed(self, event: ProcessingFailedEvent):
        self.failed_count += 1
        print(f"Processing failed: {event.document_id}")
        print(f"Error: {event.error_message}")

# Register handlers
handler = ProcessingEventHandler()
event_publisher.subscribe(handler.handle_processed)
event_publisher.subscribe(handler.handle_failed)
```

## Performance Optimization

### Memory Management

```python
import gc
from jtext.core import get_file_size_mb

def process_large_files(file_paths: list[str]) -> None:
    """Process large files with memory management."""

    for file_path in file_paths:
        # Check file size
        file_size = get_file_size_mb(file_path)
        if file_size > 100:  # Large file
            print(f"Processing large file: {file_size}MB")

        # Process file
        result = use_case.execute(request)

        # Force garbage collection for large files
        if file_size > 50:
            gc.collect()
```

### Concurrent Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

async def process_files_concurrently(file_paths: List[str]) -> List[Result]:
    """Process multiple files concurrently."""

    def process_single_file(file_path: str):
        request = ProcessDocumentRequest(
            file_path=file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE
        )
        return use_case.execute(request)

    # Use thread pool for CPU-bound operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_single_file, file_path)
            for file_path in file_paths
        ]

        results = await asyncio.gather(*tasks)
        return results

# Usage
file_paths = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
results = asyncio.run(process_files_concurrently(file_paths))

for i, result in enumerate(results):
    if result.is_ok:
        print(f"✅ File {i+1} processed successfully")
    else:
        print(f"❌ File {i+1} failed: {result.unwrap_err()}")
```

### Caching Results

```python
from functools import lru_cache
from jtext.core import get_file_hash

class CachedProcessingService:
    def __init__(self, use_case: ProcessDocumentUseCase):
        self.use_case = use_case
        self.cache = {}

    def process_with_cache(self, request: ProcessDocumentRequest) -> Result:
        # Create cache key from file hash
        file_hash = get_file_hash(request.file_path)
        cache_key = f"{file_hash}_{request.language.value}_{request.enable_correction}"

        # Check cache
        if cache_key in self.cache:
            print("Cache hit!")
            return self.cache[cache_key]

        # Process and cache result
        result = self.use_case.execute(request)
        if result.is_ok:
            self.cache[cache_key] = result

        return result

# Usage
cached_service = CachedProcessingService(use_case)
result = cached_service.process_with_cache(request)
```

## Best Practices

### 1. Always Handle Errors

```python
# ✅ Good: Handle both success and error cases
result = use_case.execute(request)
if result.is_ok:
    response = result.unwrap()
    # Handle success
else:
    error = result.unwrap_err()
    logger.error(f"Processing failed: {error}")
    # Handle error appropriately

# ❌ Bad: Assume success
response = use_case.execute(request).unwrap()  # May crash
```

### 2. Use Correlation IDs

```python
# ✅ Good: Set correlation ID for request tracing
correlation_id = generate_id()
set_correlation_id(correlation_id)

logger.info("Starting document processing")
result = use_case.execute(request)
logger.info("Document processing completed")
```

### 3. Validate Inputs

```python
# ✅ Good: Validate before processing
from jtext.core import is_supported_file, validate_file_path

def safe_process(file_path: str) -> Result[str, str]:
    # Validate file path
    if not validate_file_path(file_path):
        return Err("Invalid file path")

    # Check if file type is supported
    if not is_supported_file(file_path):
        return Err("Unsupported file type")

    # Proceed with processing
    return process_document(file_path)
```

### 4. Monitor Performance

```python
# ✅ Good: Monitor critical operations
with PerformanceMonitor("critical_operation") as monitor:
    result = expensive_operation()
    monitor.add_metric("items_processed", len(items))

# Review performance logs for optimization
```

### 5. Handle Resource Cleanup

```python
# ✅ Good: Proper resource management
try:
    service = ExternalService()
    result = service.process(data)
finally:
    service.cleanup()  # Always cleanup

# Or use context managers when available
with ExternalService() as service:
    result = service.process(data)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Problem: ImportError for optional dependencies
try:
    from jtext.infrastructure import TesseractOCRService
    ocr_service = TesseractOCRService()
except ImportError as e:
    print(f"OCR not available: {e}")
    print("Install with: pip install pytesseract pillow")
```

#### 2. File Not Found

```python
# Problem: FilePath validation fails
try:
    file_path = FilePath("nonexistent.jpg")
except ValueError as e:
    print(f"File validation failed: {e}")
    # Check if file exists before creating FilePath
```

#### 3. Service Unavailable

```python
# Problem: External service not running
result = llm_service.correct_text(text, language)
if result.is_err:
    error = result.unwrap_err()
    if "connection" in error.lower():
        print("LLM service not running. Start with: ollama serve")
```

#### 4. Memory Issues

```python
# Problem: Out of memory with large files
file_size = get_file_size_mb(file_path)
if file_size > 100:
    print(f"Warning: Large file ({file_size}MB). Consider batch processing.")
```

For more troubleshooting help, check the logs and enable debug logging:

```python
import logging
logging.getLogger("jtext").setLevel(logging.DEBUG)
```
