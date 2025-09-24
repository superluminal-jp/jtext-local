# Development Guide

Welcome to the jtext development guide! This document will help you get started contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Adding New Features](#adding-new-features)
- [Debugging](#debugging)
- [Performance Guidelines](#performance-guidelines)
- [Contributing](#contributing)

## Getting Started

### Prerequisites

- Python 3.8+ (recommended: 3.11+)
- Git
- Basic understanding of Clean Architecture and Domain-Driven Design
- Familiarity with Python type hints and modern Python practices

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd jtext-local

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Development Environment Setup

### IDE Configuration

#### VS Code (Recommended)

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm

1. Set interpreter to `./venv/bin/python`
2. Enable mypy inspection
3. Configure Black as external tool
4. Set pytest as test runner

### Dependencies

#### Core Dependencies

```bash
# Runtime dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

#### Optional Dependencies

```bash
# OCR capabilities
pip install pytesseract pillow

# Audio transcription
pip install faster-whisper

# LLM integration
pip install requests

# Additional development tools
pip install ipython jupyter
```

### Environment Variables

Create `.env` file:

```bash
# Logging
JTEXT_LOG_LEVEL=DEBUG
JTEXT_LOG_FILE=logs/jtext.log

# Services
JTEXT_OCR_LANGUAGE=jpn+eng
JTEXT_ASR_MODEL=base
JTEXT_LLM_MODEL=llama3.2:3b
JTEXT_OLLAMA_BASE_URL=http://localhost:11434

# Development
JTEXT_ENV=development
JTEXT_DEBUG=true
```

## Project Structure

### Layer Organization

```
jtext/
├── core.py                    # Shared utilities
├── interface/                 # Interface Layer
│   ├── __init__.py
│   └── cli.py               # Command-line interface
├── application/               # Application Layer
│   ├── __init__.py
│   ├── dto.py               # Data Transfer Objects
│   └── use_cases.py         # Business use cases
├── domain/                    # Domain Layer
│   ├── __init__.py
│   ├── value_objects.py     # Value objects
│   ├── entities.py          # Domain entities
│   ├── events.py            # Domain events
│   ├── services.py          # Domain services
│   ├── repositories.py      # Repository interfaces
│   └── interfaces.py        # Service interfaces
└── infrastructure/           # Infrastructure Layer
    ├── __init__.py
    ├── errors.py            # Error handling
    ├── logging.py           # Structured logging
    ├── repositories.py      # Repository implementations
    └── services.py          # External service implementations
```

### File Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Import Organization

```python
# Standard library imports
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict

# Third-party imports
import click
import pytest

# Local imports
from jtext.core import Result, Ok, Err
from jtext.domain import Document, DocumentId
from jtext.application import ProcessDocumentUseCase
from jtext.infrastructure import InMemoryDocumentRepository
```

## Development Workflow

### 1. Feature Development Process

```bash
# 1. Create feature branch
git checkout -b feature/add-new-processing-service

# 2. Write failing tests (TDD)
# Edit tests/test_new_feature.py

# 3. Run tests to confirm they fail
pytest tests/test_new_feature.py

# 4. Implement minimum code to pass tests
# Edit jtext/

# 5. Run tests to confirm they pass
pytest tests/test_new_feature.py

# 6. Refactor and improve
# Improve implementation while keeping tests passing

# 7. Run full test suite
pytest

# 8. Run quality checks
make quality-check

# 9. Commit changes
git add .
git commit -m "feat: add new processing service"

# 10. Push and create PR
git push origin feature/add-new-processing-service
```

### 2. Making Changes

#### Domain Changes

```bash
# 1. Start with domain tests
# tests/test_domain.py

# 2. Implement domain logic
# jtext/domain/

# 3. Update application layer
# jtext/application/

# 4. Update infrastructure if needed
# jtext/infrastructure/

# 5. Update interface if needed
# jtext/interface/
```

#### Infrastructure Changes

```bash
# 1. Add interface to domain layer
# jtext/domain/interfaces.py

# 2. Implement in infrastructure
# jtext/infrastructure/services.py

# 3. Add tests
# tests/test_infrastructure.py

# 4. Wire up in application
# jtext/application/use_cases.py
```

### 3. Common Development Tasks

#### Adding a New Value Object

```python
# 1. Add to jtext/domain/value_objects.py
@dataclass(frozen=True)
class NewValueObject:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Value cannot be empty")

# 2. Add to jtext/domain/__init__.py
from .value_objects import NewValueObject

__all__ = [..., "NewValueObject"]

# 3. Add tests to tests/test_domain.py
def test_new_value_object_creation():
    obj = NewValueObject("test")
    assert obj.value == "test"

def test_new_value_object_validation():
    with pytest.raises(ValueError):
        NewValueObject("")
```

#### Adding a New Use Case

```python
# 1. Add to jtext/application/use_cases.py
class NewUseCase:
    def __init__(self, repository: SomeRepository):
        self.repository = repository

    def execute(self, request: NewRequest) -> Result[NewResponse, str]:
        # Implementation
        pass

# 2. Add DTO if needed
@dataclass
class NewRequest:
    param: str

@dataclass
class NewResponse:
    result: str

# 3. Add tests
def test_new_use_case():
    repository = Mock()
    use_case = NewUseCase(repository)

    request = NewRequest("test")
    result = use_case.execute(request)

    assert result.is_ok
```

#### Adding a New Service

```python
# 1. Add interface to jtext/domain/interfaces.py
class NewService(ABC):
    @abstractmethod
    def process(self, data: str) -> Result[str, str]:
        pass

# 2. Implement in jtext/infrastructure/services.py
class ConcreteNewService(NewService):
    def __init__(self):
        self.logger = get_logger("ConcreteNewService")

    def process(self, data: str) -> Result[str, str]:
        try:
            # Implementation
            return Ok("processed")
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return Err(str(e))

# 3. Add tests with mocking
@patch("jtext.infrastructure.services.external_dependency")
def test_concrete_new_service(mock_external):
    mock_external.return_value = "mocked_result"

    service = ConcreteNewService()
    result = service.process("test")

    assert result.is_ok
    assert result.unwrap() == "processed"
```

## Code Standards

### 1. Type Hints

```python
# ✅ Good: Complete type annotations
from typing import Optional, List, Dict, Union

def process_documents(
    file_paths: List[str],
    options: Optional[Dict[str, str]] = None
) -> Result[List[ProcessingResult], str]:
    pass

# ❌ Bad: Missing type hints
def process_documents(file_paths, options=None):
    pass
```

### 2. Error Handling

```python
# ✅ Good: Use Result types
def risky_operation() -> Result[str, str]:
    try:
        result = external_call()
        return Ok(result)
    except SpecificException as e:
        return Err(f"Specific error: {e}")
    except Exception as e:
        return Err(f"Unexpected error: {e}")

# ❌ Bad: Unhandled exceptions
def risky_operation() -> str:
    return external_call()  # May raise exception
```

### 3. Logging

```python
# ✅ Good: Structured logging with context
logger = get_logger("ServiceName")

def process_file(file_path: str) -> None:
    correlation_id = generate_id()
    set_correlation_id(correlation_id)

    logger.info("Processing started", extra={
        "file_path": file_path,
        "file_size": get_file_size_mb(file_path)
    })

    try:
        # Processing logic
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error("Processing failed", extra={"error": str(e)})

# ❌ Bad: Print statements or basic logging
def process_file(file_path: str) -> None:
    print(f"Processing {file_path}")  # Not structured
    logging.info("Processing started")  # No context
```

### 4. Documentation

```python
# ✅ Good: Comprehensive docstrings
def process_document(
    file_path: str,
    language: Language,
    enable_correction: bool = False
) -> Result[ProcessingResult, str]:
    """
    Process a document using OCR or ASR based on file type.

    Args:
        file_path: Path to the document file
        language: Target language for processing
        enable_correction: Whether to apply LLM correction

    Returns:
        Result containing ProcessingResult on success or error message on failure

    Raises:
        None (uses Result type for error handling)

    Example:
        >>> result = process_document("doc.jpg", Language.JAPANESE, True)
        >>> if result.is_ok:
        ...     print(f"Processed: {result.unwrap().content}")
    """
    pass

# ❌ Bad: Missing or incomplete documentation
def process_document(file_path, language, enable_correction=False):
    # Process document
    pass
```

### 5. Domain Language

```python
# ✅ Good: Use ubiquitous language
class Document:
    def update_processing_status(self, status: ProcessingStatus) -> None:
        """Update the document's processing status."""
        pass

class DocumentProcessingService:
    def can_process_document(self, document: Document) -> bool:
        """Check if document can be processed based on business rules."""
        pass

# ❌ Bad: Technical language that doesn't match domain
class Doc:
    def set_flag(self, flag: int) -> None:
        pass

class DataProcessor:
    def check_valid(self, data: Any) -> bool:
        pass
```

## Testing Guidelines

### 1. Test Structure

```python
# Arrange-Act-Assert pattern
def test_document_processing():
    # Arrange
    document = Document(
        id=DocumentId.generate(),
        file_path=FilePath("test.jpg"),
        # ... other fields
    )

    # Act
    result = service.process(document)

    # Assert
    assert result.is_ok
    assert result.unwrap().confidence.value > 0.8
```

### 2. Test Categories

#### Unit Tests

```python
# Test individual components in isolation
def test_document_id_generation():
    doc_id = DocumentId.generate()
    assert doc_id.value.startswith("doc_")
    assert len(doc_id.value) > 4

def test_confidence_validation():
    with pytest.raises(ValueError):
        Confidence(-0.1)  # Invalid confidence
```

#### Integration Tests

```python
# Test component interactions
def test_document_processing_use_case():
    # Setup dependencies
    repo = InMemoryDocumentRepository()
    publisher = EventPublisherService()
    use_case = ProcessDocumentUseCase(repo, publisher)

    # Test integration
    request = ProcessDocumentRequest(...)
    result = use_case.execute(request)

    assert result.is_ok
    # Verify side effects
    assert len(repo.find_all().unwrap()) == 1
```

### 3. Mocking Guidelines

```python
# Mock external dependencies
@patch("jtext.infrastructure.services.pytesseract.image_to_string")
def test_ocr_service(mock_tesseract):
    # Setup mock
    mock_tesseract.return_value = "extracted text"

    # Test
    service = TesseractOCRService()
    result = service.extract_text("test.jpg", Language.JAPANESE)

    # Verify
    assert result.is_ok
    assert result.unwrap().content == "extracted text"
    mock_tesseract.assert_called_once()
```

### 4. Test Data Management

```python
# Use fixtures for reusable test data
@pytest.fixture
def sample_document():
    return Document(
        id=DocumentId.generate(),
        file_path=FilePath("test.jpg"),
        document_type=DocumentType.IMAGE,
        language=Language.JAPANESE,
        status=ProcessingStatus.PENDING,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def test_document_processing(sample_document):
    # Use fixture
    assert sample_document.status == ProcessingStatus.PENDING
```

## Adding New Features

### 1. Feature Planning

Before implementing:

1. **Understand the domain**: Where does this feature fit in the business domain?
2. **Design the interface**: What should the API look like?
3. **Consider dependencies**: What external services or data are needed?
4. **Plan tests**: How will you verify the feature works?

### 2. Implementation Order

Follow the dependency direction:

```
1. Domain Layer (entities, value objects, interfaces)
   ↓
2. Application Layer (use cases, DTOs)
   ↓
3. Infrastructure Layer (service implementations)
   ↓
4. Interface Layer (CLI commands, API endpoints)
```

### 3. Example: Adding Vision Analysis

#### Step 1: Domain Layer

```python
# jtext/domain/value_objects.py
@dataclass(frozen=True)
class VisionPrompt:
    text: str

    def __post_init__(self) -> None:
        if not self.text or len(self.text) > 1000:
            raise ValueError("Prompt must be 1-1000 characters")

# jtext/domain/entities.py
@dataclass
class VisionResult(ProcessingResult):
    description: str
    detected_objects: List[str]

# jtext/domain/interfaces.py
class VisionService(ABC):
    @abstractmethod
    def analyze_image(
        self,
        image_path: str,
        prompt: VisionPrompt
    ) -> Result[VisionResult, str]:
        pass
```

#### Step 2: Application Layer

```python
# jtext/application/dto.py
@dataclass
class VisionAnalysisRequest:
    file_path: str
    prompt: str

@dataclass
class VisionAnalysisResponse:
    description: str
    objects: List[str]
    confidence: float

# jtext/application/use_cases.py
class AnalyzeImageUseCase:
    def __init__(self, vision_service: VisionService):
        self.vision_service = vision_service

    def execute(
        self,
        request: VisionAnalysisRequest
    ) -> Result[VisionAnalysisResponse, str]:
        # Implementation
        pass
```

#### Step 3: Infrastructure Layer

```python
# jtext/infrastructure/services.py
class OllamaVisionService(VisionService):
    def __init__(self, model_name: str = "llava:7b"):
        self.model_name = model_name
        self.logger = get_logger("OllamaVisionService")

    def analyze_image(
        self,
        image_path: str,
        prompt: VisionPrompt
    ) -> Result[VisionResult, str]:
        # Implementation with error handling
        pass
```

#### Step 4: Interface Layer

```python
# jtext/interface/cli.py
@cli.command()
@click.argument('image_path')
@click.option('--prompt', default='Describe this image')
def analyze(image_path: str, prompt: str):
    """Analyze image with vision model."""
    # Implementation
    pass
```

#### Step 5: Tests

```python
# tests/test_vision_feature.py
class TestVisionFeature:
    def test_vision_prompt_validation(self):
        # Test value object
        pass

    def test_analyze_image_use_case(self):
        # Test use case with mocked service
        pass

    @patch("jtext.infrastructure.services.requests.post")
    def test_ollama_vision_service(self, mock_post):
        # Test service implementation
        pass
```

## Debugging

### 1. Logging Configuration

```python
# Enable debug logging
import logging
logging.getLogger("jtext").setLevel(logging.DEBUG)

# Or via environment
export JTEXT_LOG_LEVEL=DEBUG
```

### 2. Common Debug Scenarios

#### Service Integration Issues

```python
# Check service availability
try:
    service = TesseractOCRService()
    print("✅ Service available")
except ImportError as e:
    print(f"❌ Service unavailable: {e}")
    print("Install dependencies: pip install pytesseract pillow")
```

#### File Processing Issues

```python
# Debug file operations
from jtext.core import validate_file_path, is_supported_file, get_file_size_mb

file_path = "problem_file.jpg"

print(f"File exists: {validate_file_path(file_path)}")
print(f"File supported: {is_supported_file(file_path)}")
print(f"File size: {get_file_size_mb(file_path)}MB")
```

#### Result Type Issues

```python
# Debug Result chains
result = (
    validate_input(data)
    .and_then(lambda x: print(f"Validated: {x}") or Ok(x))  # Debug print
    .and_then(process_data)
    .and_then(lambda x: print(f"Processed: {x}") or Ok(x))  # Debug print
    .map(format_output)
)

if result.is_err:
    print(f"Pipeline failed: {result.unwrap_err()}")
```

### 3. Performance Debugging

```python
# Use performance monitoring
with PerformanceMonitor("slow_operation") as monitor:
    result = slow_operation()
    monitor.add_metric("items_processed", len(items))

# Check metrics in logs or implement custom metric collection
```

## Performance Guidelines

### 1. Memory Management

```python
# ✅ Good: Process large files in chunks
def process_large_file(file_path: str) -> Result[str, str]:
    file_size = get_file_size_mb(file_path)

    if file_size > 100:  # Large file
        # Use streaming processing
        return process_in_chunks(file_path)
    else:
        # Normal processing
        return process_normally(file_path)

# ✅ Good: Clean up resources
try:
    resource = acquire_resource()
    result = process_with_resource(resource)
finally:
    resource.cleanup()
```

### 2. Async Processing

```python
# For I/O bound operations
import asyncio

async def process_multiple_files(file_paths: List[str]) -> List[Result]:
    tasks = [process_file_async(path) for path in file_paths]
    return await asyncio.gather(*tasks)
```

### 3. Caching

```python
# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(data: str) -> str:
    # Expensive computation
    return result
```

## Contributing

### 1. Pull Request Process

1. **Fork and branch**: Create feature branch from main
2. **Implement**: Follow TDD and clean architecture principles
3. **Test**: Ensure all tests pass and coverage is maintained
4. **Document**: Update relevant documentation
5. **Quality**: Run quality checks and fix issues
6. **Submit**: Create PR with clear description

### 2. PR Requirements

- [ ] Tests pass (`pytest`)
- [ ] Quality checks pass (`make quality-check`)
- [ ] Code coverage maintained (>65%)
- [ ] Documentation updated
- [ ] Clean commit history
- [ ] Clear PR description

### 3. Code Review Guidelines

When reviewing code:

- **Architecture**: Does it follow Clean Architecture principles?
- **Domain**: Does it use ubiquitous language correctly?
- **Testing**: Are there adequate tests?
- **Error Handling**: Are errors handled properly with Result types?
- **Performance**: Are there any obvious performance issues?
- **Documentation**: Is the code self-documenting with good docstrings?

### 4. Git Conventions

#### Commit Messages

```
feat: add new vision analysis service
fix: resolve OCR confidence calculation bug
docs: update API usage examples
test: add integration tests for document processing
refactor: extract common validation logic
style: format code with black
chore: update dependencies
```

#### Branch Names

```
feature/add-vision-analysis
fix/ocr-confidence-bug
docs/update-api-guide
refactor/extract-validation
```

## Best Practices Summary

### Do's ✅

- Follow Clean Architecture layer boundaries
- Use ubiquitous language consistently
- Write tests first (TDD)
- Handle errors with Result types
- Use structured logging with correlation IDs
- Document public APIs thoroughly
- Keep functions small and focused
- Use type hints everywhere
- Validate inputs early
- Monitor performance of critical operations

### Don'ts ❌

- Don't violate layer dependencies (no upward dependencies)
- Don't use technical language in domain layer
- Don't ignore test failures
- Don't use bare exceptions or print statements
- Don't hardcode configuration values
- Don't create large, monolithic functions
- Don't forget to handle edge cases
- Don't skip documentation updates
- Don't commit without running quality checks
- Don't mix business logic with infrastructure concerns

## Getting Help

### Resources

- **Architecture Documentation**: `docs/ARCHITECTURE.md`
- **API Usage Guide**: `docs/API_USAGE.md`
- **Code Examples**: `examples/` directory
- **Test Examples**: `tests/` directory

### Common Commands

```bash
# Setup
make setup              # Initial project setup
make install-dev       # Install development dependencies

# Development
make test              # Run tests
make quality-check     # Run all quality checks
make format           # Format code
make lint             # Run linters
make type-check       # Run type checking

# Documentation
make docs             # Generate documentation
make docs-serve       # Serve documentation locally

# Clean
make clean            # Clean temporary files
make clean-all        # Clean everything including venv
```

### Troubleshooting

If you encounter issues:

1. **Check dependencies**: `pip list` and compare with requirements
2. **Verify Python version**: Ensure Python 3.8+
3. **Clean and reinstall**: `make clean && make setup`
4. **Check logs**: Enable debug logging
5. **Run isolated tests**: `pytest tests/test_specific.py -v`

For more help, check existing issues or create a new one with:

- Python version
- Operating system
- Full error message
- Steps to reproduce
