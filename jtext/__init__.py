"""
jtext - Japanese Text Processing System

A high-precision local text extraction system for Japanese documents,
images, and audio using OCR, ASR, and LLM correction.

This package follows Clean Architecture principles with clear layer separation:
- interface/: User interfaces and adapters (CLI, Web, etc.)
- application/: Use cases and DTOs
- domain/: Core business logic and entities
- infrastructure/: External services and implementations
- core.py: Shared utilities and common functionality
"""

__version__ = "1.0.0"
__author__ = "jtext Development Team"
__email__ = "dev@jtext.local"

# Core utilities
from .core import (
    Result,
    Ok,
    Err,
    generate_id,
    get_file_hash,
    get_file_size_mb,
    is_supported_file,
    validate_file_path,
    validate_output_directory,
    setup_logging,
    PerformanceMonitor,
    AppConfig,
)

# Domain layer
from .domain import (
    # Value Objects
    DocumentId,
    ProcessingStatus,
    DocumentType,
    Language,
    FilePath,
    Confidence,
    ProcessingMetrics,
    # Events
    DomainEvent,
    DocumentProcessedEvent,
    ProcessingFailedEvent,
    # Entities
    Document,
    ImageDocument,
    AudioDocument,
    ProcessingResult,
    OCRResult,
    TranscriptionResult,
    # Services
    DocumentProcessingService,
    # Interfaces
    DocumentRepository,
    ProcessingResultRepository,
    OCRService,
    TranscriptionService,
    CorrectionService,
    EventPublisher,
)

# Application layer
from .application import (
    # DTOs
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    DocumentDTO,
    ProcessingResultDTO,
    ProcessingStatisticsDTO,
    # Use Cases
    UseCase,
    ProcessDocumentUseCase,
    ProcessImageUseCase,
    ProcessAudioUseCase,
    GetProcessingResultUseCase,
    ListDocumentsUseCase,
    GetProcessingStatisticsUseCase,
)

# Infrastructure layer
from .infrastructure import (
    # Error Handling
    ProcessingError,
    ValidationError,
    InfrastructureError,
    DomainError,
    ExternalServiceError,
    SystemError,
    ErrorHandler,
    CircuitBreaker,
    # Logging
    StructuredLogger,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    # Repositories
    InMemoryDocumentRepository,
    InMemoryProcessingResultRepository,
    # Services
    TesseractOCRService,
    WhisperTranscriptionService,
    OllamaCorrectionService,
    EventPublisherService,
)

# Interface layer
from .interface import CLI, create_cli

# Create default CLI instance
cli = create_cli()

__all__ = [
    # Core utilities
    "Result",
    "Ok",
    "Err",
    "generate_id",
    "get_file_hash",
    "get_file_size_mb",
    "is_supported_file",
    "validate_file_path",
    "validate_output_directory",
    "setup_logging",
    "PerformanceMonitor",
    "AppConfig",
    # Domain
    "DocumentId",
    "ProcessingStatus",
    "DocumentType",
    "Language",
    "FilePath",
    "Confidence",
    "ProcessingMetrics",
    "DomainEvent",
    "DocumentProcessedEvent",
    "ProcessingFailedEvent",
    "Document",
    "ImageDocument",
    "AudioDocument",
    "ProcessingResult",
    "OCRResult",
    "TranscriptionResult",
    "DocumentProcessingService",
    "DocumentRepository",
    "ProcessingResultRepository",
    "OCRService",
    "TranscriptionService",
    "CorrectionService",
    "EventPublisher",
    # Application
    "ProcessDocumentRequest",
    "ProcessDocumentResponse",
    "DocumentDTO",
    "ProcessingResultDTO",
    "ProcessingStatisticsDTO",
    "UseCase",
    "ProcessDocumentUseCase",
    "ProcessImageUseCase",
    "ProcessAudioUseCase",
    "GetProcessingResultUseCase",
    "ListDocumentsUseCase",
    "GetProcessingStatisticsUseCase",
    # Infrastructure
    "ProcessingError",
    "ValidationError",
    "InfrastructureError",
    "DomainError",
    "ExternalServiceError",
    "SystemError",
    "ErrorHandler",
    "CircuitBreaker",
    "StructuredLogger",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "InMemoryDocumentRepository",
    "InMemoryProcessingResultRepository",
    "TesseractOCRService",
    "WhisperTranscriptionService",
    "OllamaCorrectionService",
    "EventPublisherService",
    # Interface
    "CLI",
    "create_cli",
    "cli",
]
