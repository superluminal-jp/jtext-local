# Architecture Documentation

## Overview

jtext follows **Clean Architecture** principles combined with **Domain-Driven Design (DDD)** patterns to create a maintainable, testable, and scalable Japanese text processing system.

## Architectural Principles

### 1. Clean Architecture

- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Layer Separation**: Clear boundaries between interface, application, domain, and infrastructure
- **Framework Independence**: Business logic is independent of external frameworks
- **Testability**: Each layer can be tested independently

### 2. Domain-Driven Design (DDD)

- **Ubiquitous Language**: Consistent terminology across the codebase
- **Rich Domain Model**: Entities and value objects encapsulate business logic
- **Domain Events**: Communicate important business occurrences
- **Bounded Contexts**: Clear boundaries around related functionality

### 3. Independent Components

- **Self-Contained Modules**: Each component has minimal dependencies
- **Clear Interfaces**: Well-defined contracts between components
- **Easy to Understand**: Logical organization that matches business concepts

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │     CLI     │  │   Web API   │  │   GraphQL   │ (Future)│
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Dependency: Uses
┌─────────────────────▼───────────────────────────────────────┐
│                 Application Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Use Cases │  │     DTOs    │  │ App Services│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │ Dependency: Uses
┌─────────────────────▼───────────────────────────────────────┐
│                   Domain Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Entities   │  │Value Objects│  │   Events    │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │  Services   │  │Repositories │  │ Interfaces  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────▲───────────────────────────────────────┘
                      │ Dependency: Implements
┌─────────────────────┴───────────────────────────────────────┐
│                Infrastructure Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │External APIs│  │ Repositories│  │Error Handling│        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │   Logging   │  │Circuit Breaker│ │   Services  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Core Module (`core.py`)

**Purpose**: Shared utilities and common functionality across all layers.

**Key Components**:

- **Result Type**: Railway-Oriented Programming implementation
- **Utility Functions**: File operations, validation, ID generation
- **Performance Monitor**: Context manager for performance tracking
- **App Configuration**: Configuration management with validation
- **Constants**: Application-wide constants and defaults

```python
# Result Type Usage
from jtext.core import Result, Ok, Err

def process_file(path: str) -> Result[str, str]:
    if not os.path.exists(path):
        return Err("File not found")
    return Ok("File processed successfully")

# Performance Monitoring
from jtext.core import PerformanceMonitor

with PerformanceMonitor("file_processing") as monitor:
    # Processing logic here
    pass
# Automatically logs performance metrics
```

### Interface Layer (`interface/`)

**Purpose**: Adapters that convert external requests into internal use case calls.

**Components**:

- **CLI (`cli.py`)**: Command-line interface using Click framework
- **Future**: Web API, GraphQL endpoints

**Responsibilities**:

- Input validation and parsing
- Request/response conversion
- Error presentation to users
- Service dependency injection

```python
# CLI Command Example
@click.command()
@click.argument('file_path')
@click.option('--language', default='jpn+eng')
def ocr(file_path: str, language: str):
    """Process image with OCR."""
    # Convert CLI args to domain objects
    request = ProcessDocumentRequest(
        file_path=file_path,
        document_type=DocumentType.IMAGE,
        language=Language(language)
    )

    # Execute use case
    result = use_case.execute(request)

    # Present result to user
    if result.is_ok:
        click.echo(f"Success: {result.unwrap().document_id}")
    else:
        click.echo(f"Error: {result.unwrap_err()}")
```

### Application Layer (`application/`)

**Purpose**: Orchestrates domain objects to perform business operations.

**Components**:

- **DTOs (`dto.py`)**: Data transfer objects for cross-boundary communication
- **Use Cases (`use_cases.py`)**: Business operation implementations

**Key Use Cases**:

- `ProcessDocumentUseCase`: Main document processing workflow
- `GetProcessingResultUseCase`: Retrieve processing results
- `ListDocumentsUseCase`: List processed documents
- `GetProcessingStatisticsUseCase`: Processing analytics

```python
# Use Case Example
class ProcessDocumentUseCase:
    def __init__(
        self,
        document_repository: DocumentRepository,
        result_repository: ProcessingResultRepository,
        event_publisher: EventPublisher
    ):
        self.document_repository = document_repository
        self.result_repository = result_repository
        self.event_publisher = event_publisher

    def execute(self, request: ProcessDocumentRequest) -> Result[ProcessDocumentResponse, str]:
        with PerformanceMonitor("process_document") as monitor:
            # 1. Create domain entity
            document = self._create_document(request)

            # 2. Save document
            save_result = self.document_repository.save(document)
            if save_result.is_err:
                return Err(f"Failed to save document: {save_result.unwrap_err()}")

            # 3. Process based on type
            if request.document_type == DocumentType.IMAGE:
                return self._process_image_document(document, request)
            elif request.document_type == DocumentType.AUDIO:
                return self._process_audio_document(document, request)
            else:
                return Err(f"Unsupported document type: {request.document_type}")
```

### Domain Layer (`domain/`)

**Purpose**: Core business logic and domain concepts.

**Components**:

#### Value Objects (`value_objects.py`)

Immutable objects that describe domain concepts:

- `DocumentId`: Unique document identifier
- `Confidence`: Processing confidence (0.0-1.0)
- `FilePath`: Validated file path
- `ProcessingMetrics`: Performance and quality metrics

#### Entities (`entities.py`)

Objects with identity and lifecycle:

- `Document`: Base document entity
- `ImageDocument`: Image-specific document
- `AudioDocument`: Audio-specific document
- `ProcessingResult`: OCR/ASR result

#### Domain Events (`events.py`)

Important business occurrences:

- `DocumentProcessedEvent`: Document processing completed
- `ProcessingFailedEvent`: Processing failed

#### Services (`services.py`)

Complex domain logic that doesn't belong to entities:

- `DocumentProcessingService`: Processing capability checks

#### Interfaces (`repositories.py`, `interfaces.py`)

Abstract contracts for external dependencies:

- Repository interfaces for data access
- Service interfaces for external operations

```python
# Value Object Example
@dataclass(frozen=True)
class DocumentId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or len(self.value) > 255:
            raise ValueError("DocumentId must be non-empty and ≤255 characters")

    @classmethod
    def generate(cls) -> "DocumentId":
        return cls(value=f"doc_{generate_id()}")

# Entity Example
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
        """Update document processing status."""
        self.status = new_status
        self.updated_at = datetime.now()
```

### Infrastructure Layer (`infrastructure/`)

**Purpose**: Technical concerns and external service integrations.

**Components**:

#### Error Handling (`errors.py`)

- Custom exception types
- Circuit breaker implementation
- Error recovery patterns

#### Logging (`logging.py`)

- Structured logging with correlation IDs
- OpenTelemetry-compatible format
- Context propagation

#### Repositories (`repositories.py`)

- In-memory implementations
- Future: Database implementations

#### Services (`services.py`)

- External service integrations
- OCR, ASR, LLM service implementations
- Optional dependency handling

```python
# Service Implementation Example
class TesseractOCRService(OCRService):
    def __init__(self):
        if not HAS_TESSERACT:
            raise ImportError("Tesseract dependencies not available")
        self.logger = get_logger("TesseractOCRService")

    def extract_text(self, image_path: str, language: Language) -> Result[OCRResult, str]:
        if not HAS_TESSERACT:
            return Err("Tesseract dependencies not available")

        try:
            # Configure Tesseract
            config = f"--oem 3 --psm 6 -l {language.value}"

            # Extract text
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config=config)

            # Get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Create result
            return Ok(OCRResult(
                result_type="OCR",
                content=text.strip(),
                confidence=Confidence.from_percentage(avg_confidence),
                processing_time=2.0,  # Mock timing
                created_at=datetime.now()
            ))

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return Err(f"OCR processing failed: {str(e)}")
```

## Data Flow

### Document Processing Flow

```
1. CLI Input → ProcessDocumentRequest (DTO)
                      ↓
2. ProcessDocumentUseCase.execute()
                      ↓
3. Create Document Entity (Domain)
                      ↓
4. Save Document (Repository)
                      ↓
5. Choose Processing Strategy (Domain Service)
                      ↓
6. Call External Service (Infrastructure)
                      ↓
7. Create ProcessingResult (Domain)
                      ↓
8. Save Result (Repository)
                      ↓
9. Publish Domain Event
                      ↓
10. Return ProcessDocumentResponse (DTO)
                      ↓
11. CLI Output
```

### Error Handling Flow

```
1. External Service Failure
                      ↓
2. Service Returns Err(error_message)
                      ↓
3. Use Case Handles Error
                      ↓
4. Logs Error with Correlation ID
                      ↓
5. Returns Err to Interface Layer
                      ↓
6. Interface Presents Error to User
```

## Design Patterns

### Railway-Oriented Programming

Used throughout for error handling:

```python
def process_pipeline(input: str) -> Result[str, str]:
    return (
        validate_input(input)
        .and_then(transform_data)
        .and_then(save_to_storage)
        .map(format_output)
    )
```

### Domain Events

For decoupled communication:

```python
# Publish event after processing
event = DocumentProcessedEvent(
    event_id=generate_id(),
    document_id=document.id.value,
    processing_result=result.to_dict(),
    occurred_at=datetime.now()
)
self.event_publisher.publish(event)
```

### Repository Pattern

For data access abstraction:

```python
class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> Result[None, str]:
        pass

    @abstractmethod
    def find_by_id(self, document_id: DocumentId) -> Result[Optional[Document], str]:
        pass
```

### Dependency Injection

At the application boundary:

```python
# In CLI
def create_cli() -> CLI:
    # Infrastructure
    document_repo = InMemoryDocumentRepository()
    result_repo = InMemoryProcessingResultRepository()
    event_publisher = EventPublisherService()

    # Services
    ocr_service = TesseractOCRService()
    transcription_service = WhisperTranscriptionService()

    # Use cases
    process_use_case = ProcessDocumentUseCase(
        document_repo, result_repo, event_publisher
    )

    return CLI(process_use_case)
```

## Testing Strategy

### Layer-Specific Testing

#### Domain Layer

- **Unit Tests**: Value objects, entities, domain services
- **Focus**: Business logic correctness
- **No Dependencies**: Pure domain logic testing

#### Application Layer

- **Integration Tests**: Use cases with mocked dependencies
- **Focus**: Workflow orchestration
- **Mock External**: Repositories and services

#### Infrastructure Layer

- **Integration Tests**: External service integrations
- **Focus**: Technical implementation
- **Real/Mock Services**: Depending on test environment

#### Interface Layer

- **End-to-End Tests**: Full request/response cycles
- **Focus**: User interaction flows
- **Mock Application**: Use case mocking

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_core.py            # Core utilities
├── test_domain.py          # Domain layer
├── test_application.py     # Application layer (mock implementation)
├── test_infrastructure.py # Infrastructure layer
└── test_interface.py       # Interface layer
```

## Future Extensions

### Planned Enhancements

1. **Web API Layer**

   - RESTful API endpoints
   - GraphQL interface
   - WebSocket support for real-time updates

2. **Database Integration**

   - PostgreSQL repository implementations
   - Database migrations
   - Query optimization

3. **Event Sourcing**

   - Event store implementation
   - Event replay capabilities
   - CQRS pattern adoption

4. **Microservices**

   - Service decomposition
   - Message queue integration
   - Distributed tracing

5. **Advanced Analytics**
   - Processing metrics collection
   - Quality trend analysis
   - Performance optimization insights

### Extension Points

The architecture supports easy extension through:

- **New Use Cases**: Add to application layer
- **New Interfaces**: Implement interface adapters
- **New Services**: Implement domain interfaces
- **New Repositories**: Implement repository interfaces
- **New Events**: Add domain events for new business scenarios

## Compliance with Best Practices

### SOLID Principles

- ✅ **Single Responsibility**: Each class has one reason to change
- ✅ **Open/Closed**: Open for extension, closed for modification
- ✅ **Liskov Substitution**: Implementations are substitutable
- ✅ **Interface Segregation**: Focused, role-specific interfaces
- ✅ **Dependency Inversion**: Depend on abstractions, not concretions

### DDD Patterns

- ✅ **Ubiquitous Language**: Consistent terminology
- ✅ **Rich Domain Model**: Business logic in domain layer
- ✅ **Domain Events**: Event-driven communication
- ✅ **Repository Pattern**: Data access abstraction
- ✅ **Value Objects**: Immutable domain concepts

### Clean Architecture

- ✅ **Layer Independence**: Each layer can be tested separately
- ✅ **Framework Independence**: Domain not tied to frameworks
- ✅ **Testability**: High test coverage possible
- ✅ **Database Independence**: Domain doesn't know about storage
- ✅ **UI Independence**: Business logic separate from presentation
