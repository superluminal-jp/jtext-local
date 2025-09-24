# jtext Technical Specification

## System Overview

jtext is a high-precision Japanese text processing system built with Clean Architecture principles and Domain-Driven Design patterns. The system provides reliable text extraction from images and audio files through OCR, ASR, and LLM correction technologies.

## Architecture

### Core Principles

- **Clean Architecture**: Clear separation of concerns with dependency inversion
- **Domain-Driven Design**: Rich domain model with ubiquitous language
- **Railway-Oriented Programming**: Functional error handling with Result types
- **Independent Components**: Self-contained modules with minimal coupling
- **Test-Driven Development**: Comprehensive test coverage across all layers

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │     CLI     │  │   Web API   │  │   GraphQL   │ (Future)│
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Uses
┌─────────────────────▼───────────────────────────────────────┐
│                 Application Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Use Cases │  │     DTOs    │  │ App Services│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Uses
┌─────────────────────▼───────────────────────────────────────┐
│                   Domain Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Entities   │  │Value Objects│  │   Events    │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │  Services   │  │Repositories │  │ Interfaces  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────▲───────────────────────────────────────┘
                      │ Implements
┌─────────────────────┴───────────────────────────────────────┐
│                Infrastructure Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │External APIs│  │ Repositories│  │Error Handling│        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │   Logging   │  │Circuit Breaker│ │   Services  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Domain Model

### Value Objects

#### DocumentId

```python
@dataclass(frozen=True)
class DocumentId:
    value: str

    @classmethod
    def generate(cls) -> "DocumentId":
        return cls(value=f"doc_{generate_id()}")
```

**Purpose**: Unique identifier for documents  
**Validation**: Non-empty string, ≤255 characters  
**Generation**: Auto-generated with "doc\_" prefix

#### FilePath

```python
@dataclass(frozen=True)
class FilePath:
    value: str

    def __post_init__(self) -> None:
        if not Path(self.value).exists():
            raise ValueError(f"File does not exist: {self.value}")
```

**Purpose**: Validated file path reference  
**Validation**: File must exist on filesystem  
**Usage**: Ensures file availability before processing

#### Confidence

```python
@dataclass(frozen=True)
class Confidence:
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @classmethod
    def from_percentage(cls, percentage: float) -> "Confidence":
        return cls(value=percentage / 100.0)
```

**Purpose**: Processing confidence score  
**Range**: 0.0 to 1.0 (0% to 100%)  
**Methods**: from_float(), from_percentage(), as_percentage()

### Entities

#### Document

```python
@dataclass
class Document:
    id: DocumentId
    file_path: FilePath
    document_type: DocumentType
    language: Language
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime

    def update_status(self, new_status: ProcessingStatus) -> None:
        self.status = new_status
        self.updated_at = datetime.now()
```

**Purpose**: Core document entity with lifecycle  
**Identity**: DocumentId  
**States**: PENDING → PROCESSING → COMPLETED/FAILED  
**Operations**: Status updates with timestamp tracking

#### ProcessingResult

```python
@dataclass
class ProcessingResult:
    id: str
    document_id: DocumentId
    result_type: str
    content: str
    confidence: Confidence
    processing_time: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence.as_percentage(),
            "processing_time": self.processing_time
        }
```

**Purpose**: Processing output storage  
**Types**: OCR, ASR, CORRECTION  
**Serialization**: to_dict() for output formatting

### Domain Events

#### DocumentProcessedEvent

```python
@dataclass
class DocumentProcessedEvent(DomainEvent):
    document_id: str = ""
    processing_result: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
```

**Purpose**: Notify when document processing completes  
**Payload**: Document ID, results, processing time  
**Subscribers**: Statistics tracking, notifications, audit logging

## Application Layer

### Use Cases

#### ProcessDocumentUseCase

```python
class ProcessDocumentUseCase:
    def execute(self, request: ProcessDocumentRequest) -> Result[ProcessDocumentResponse, str]:
        with PerformanceMonitor("process_document"):
            # 1. Create document entity
            # 2. Save document
            # 3. Process based on type
            # 4. Save results
            # 5. Publish events
            # 6. Return response
```

**Purpose**: Main document processing workflow  
**Dependencies**: Repositories, event publisher  
**Flow**: Create → Save → Process → Store → Notify  
**Monitoring**: Performance tracking with correlation IDs

## Infrastructure Layer

### External Services

#### TesseractOCRService

```python
class TesseractOCRService(OCRService):
    def extract_text(self, image_path: str, language: Language) -> Result[OCRResult, str]:
        try:
            # Configure Tesseract
            config = f"--oem 3 --psm 6 -l {language.value}"

            # Extract text
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config=config)

            # Calculate confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return Ok(OCRResult(...))
        except Exception as e:
            return Err(f"OCR processing failed: {str(e)}")
```

**Technology**: Tesseract OCR Engine  
**Languages**: Japanese + English support  
**Configuration**: OEM 3, PSM 6 for document text  
**Output**: Structured OCR result with confidence

### Error Handling

#### Result Type System

```python
class Result(Generic[T, E]):
    def map(self, func: Callable[[T], U]) -> "Result[U, E]":
        if self._is_ok:
            return Ok(func(self._value))
        return Err(self._error)

    def and_then(self, func: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        if self._is_ok:
            return func(self._value)
        return Err(self._error)
```

**Pattern**: Railway-Oriented Programming  
**Operations**: map, and_then, or_else for chaining  
**Error Propagation**: Automatic error bubbling

## Performance Characteristics

### Processing Times

| Operation | File Size  | Typical Time  | Memory Usage |
| --------- | ---------- | ------------- | ------------ |
| OCR Basic | 1MB Image  | 2-3 seconds   | 100-200MB    |
| OCR + LLM | 1MB Image  | 5-8 seconds   | 200-400MB    |
| ASR Basic | 1min Audio | 10-15 seconds | 150-300MB    |

### Configuration Management

```python
@dataclass
class AppConfig:
    max_file_size_mb: float = 100.0
    max_processing_time_seconds: int = 300
    default_ocr_language: str = "jpn+eng"

    def is_file_size_valid(self, size_mb: float) -> bool:
        return size_mb <= self.max_file_size_mb
```

**Validation**: Configuration validation on startup  
**Defaults**: Sensible defaults for common use cases  
**Type Safety**: Dataclass with type hints

## Security Considerations

### Input Validation

**File Processing**:

- File type validation
- File size limits
- Path traversal prevention

**Data Privacy**:

- Local processing only
- No external data transmission
- Temporary file cleanup

## Future Enhancements

### Planned Features

**Web Interface**:

- REST API endpoints
- WebSocket for real-time updates

**Database Integration**:

- PostgreSQL repository implementations
- Migration system

**Advanced Processing**:

- Batch processing optimization
- Async processing pipeline

#### OllamaCorrectionService

```python
class OllamaCorrectionService(CorrectionService):
    def correct_text(
        self,
        text: str,
        document_type: DocumentType,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Result[str, ProcessingError], int]:
        # 1. Create correction prompt
        # 2. Call Ollama API
        # 3. Process response
        # 4. Calculate corrections applied
```

## Error Handling

### Result Pattern

The system uses Railway-Oriented Programming with the Result pattern for monadic error handling:

```python
def process_document(self, document: Document) -> Result[ProcessingResult, ProcessingError]:
    return (
        self._validate_document(document)
        .and_then(lambda doc: self._preprocess_document(doc))
        .and_then(lambda doc: self._extract_text(doc))
        .and_then(lambda result: self._apply_corrections(result))
        .and_then(lambda result: self._save_result(result))
    )
```

### Error Types

#### ProcessingError

```python
class ProcessingError(Exception):
    def __init__(
        self,
        message: str,
        error_code: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.PROCESSING,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
```

#### Specific Error Types

- `ValidationError`: Input validation failures
- `InfrastructureError`: Infrastructure service failures
- `DomainError`: Business rule violations
- `ExternalServiceError`: External API failures
- `SystemError`: System-level failures

## Logging and Observability

### Structured Logging

All logging follows OpenTelemetry standards with structured JSON output:

```python
logger.info(
    "Document processing started",
    component="DocumentProcessingService",
    operation="process_document",
    document_id=str(document.id),
    processing_time_ms=1500.0,
    memory_usage_mb=256.0,
    metadata={
        "document_type": "image",
        "language": "japanese",
        "enable_correction": True
    }
)
```

### Log Levels

- **FATAL**: System unusable (process termination)
- **ERROR**: Immediate attention required (exceptions, failures)
- **WARN**: Potentially harmful situations (deprecated APIs)
- **INFO**: General application flow (business events)
- **DEBUG**: Fine-grained debugging information
- **TRACE**: Most detailed diagnostic information

### Correlation IDs

All requests include correlation IDs for distributed tracing:

```python
correlation_id = CorrelationIdGenerator.generate()
logger.info("Request started", correlation_id=correlation_id)
```

## Testing Strategy

### Test Pyramid

1. **Unit Tests** (70%)

   - Domain entities and value objects
   - Use cases and application services
   - Repository implementations
   - Service implementations

2. **Integration Tests** (20%)

   - End-to-end use case execution
   - External service integration
   - Database operations
   - File system operations

3. **BDD Tests** (10%)
   - Business scenarios
   - User workflows
   - Error handling scenarios
   - Performance requirements

### Test Structure

#### Unit Tests

```python
class TestDocument:
    def test_document_creation(self):
        """Test Document creation with valid parameters."""
        doc_id = DocumentId.generate()
        file_path = FilePath("test.pdf")

        document = Document(
            id=doc_id,
            file_path=file_path,
            document_type=DocumentType.PDF,
            language=Language.JAPANESE,
        )

        assert document.id == doc_id
        assert document.status == ProcessingStatus.PENDING
        assert document.is_processable()
```

#### BDD Tests

```gherkin
Feature: Document Processing
  Scenario: Process a high-quality image document
    Given I have an image file "test_image.png"
    And the image contains clear Japanese text
    When I process the image with OCR
    Then the text should be extracted with confidence > 0.8
    And the processing should complete within 30 seconds
```

## Performance Requirements

### Processing Times

| Document Type  | Basic OCR | Multimodal OCR | With LLM Correction |
| -------------- | --------- | -------------- | ------------------- |
| A4 Image       | 2-5 sec   | 8-15 sec       | 15-30 sec           |
| PDF (10 pages) | 30-60 sec | 60-120 sec     | 120-240 sec         |
| Audio (30 min) | 10-15 min | N/A            | 15-20 min           |

### Accuracy Targets

| Document Type       | Basic OCR | Multimodal OCR | With LLM   |
| ------------------- | --------- | -------------- | ---------- |
| Technical Docs      | 70-80%    | 85-90%         | **95-98%** |
| Business Documents  | 75-85%    | 90-95%         | **96-99%** |
| Handwritten Text    | 60-70%    | 80-85%         | **90-95%** |
| Audio Transcription | N/A       | N/A            | **85-95%** |

### Resource Usage

- **Memory**: 2-8GB RAM (model dependent)
- **CPU**: 2-8 cores recommended
- **Storage**: 20GB+ for models and temporary files
- **Network**: Required for LLM services (Ollama)

## Security Considerations

### Data Privacy

- All processing is performed locally
- No data is sent to external services (except configured LLM)
- Temporary files are automatically cleaned up
- Logs do not contain sensitive information

### Input Validation

- File type validation
- Size limits (2GB per file)
- Path traversal protection
- Malicious file detection

### Error Handling

- No sensitive information in error messages
- Structured error responses
- Comprehensive logging for debugging
- Graceful degradation on failures

## Deployment

### System Requirements

- **OS**: macOS 12.0+ (Apple Silicon recommended)
- **Python**: 3.12+
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 20GB+ available space
- **Network**: Internet connection for model downloads

### Dependencies

#### Core Dependencies

- `click>=8.1.7`: CLI framework
- `pydantic>=2.4.2`: Data validation
- `loguru>=0.7.2`: Logging framework

#### Processing Dependencies

- `pytesseract>=0.3.10`: OCR engine
- `opencv-python>=4.8.0`: Image processing
- `pillow>=10.0.1`: Image manipulation
- `faster-whisper>=0.9.0`: Audio transcription
- `ollama>=0.1.7`: LLM integration

#### Development Dependencies

- `pytest>=7.4.3`: Testing framework
- `pytest-cov>=4.1.0`: Coverage reporting
- `black>=23.9.1`: Code formatting
- `flake8>=6.1.0`: Linting
- `mypy>=1.6.1`: Type checking

### Installation

```bash
# Development installation
git clone https://github.com/jtext/jtext-local.git
cd jtext-local
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Production installation
pip install jtext
```

## Monitoring and Observability

### Metrics

- Processing success rate
- Average processing time
- Memory usage
- CPU utilization
- Error rates by type

### Health Checks

- System resource availability
- External service connectivity
- Model availability
- Storage space

### Alerting

- High error rates
- Performance degradation
- Resource exhaustion
- Service unavailability

## Future Enhancements

### Planned Features

1. **Web Interface**: Browser-based processing interface
2. **API Server**: REST API for programmatic access
3. **Cloud Integration**: Optional cloud processing
4. **Advanced Models**: Support for newer LLM models
5. **Real-time Processing**: Live document processing

### Technical Improvements

1. **Performance**: GPU acceleration for processing
2. **Scalability**: Distributed processing support
3. **Reliability**: Enhanced error recovery
4. **Usability**: Improved user experience
5. **Integration**: Better third-party integrations
