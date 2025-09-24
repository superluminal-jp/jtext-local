"""
Infrastructure layer - External services and implementations.

This module contains infrastructure implementations including repositories,
external services, error handling, and logging following Clean Architecture.
"""

from .errors import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ProcessingError,
    ValidationError,
    InfrastructureError,
    DomainError,
    ExternalServiceError,
    SystemError,
    ErrorHandler,
    CircuitBreaker,
)
from .logging import (
    LogLevel,
    LogEntry,
    LoggingConfiguration,
    CorrelationIdGenerator,
    StructuredLogger,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    start_performance_tracking,
    get_performance_metrics,
    PerformanceMetrics,
)
from .repositories import (
    InMemoryDocumentRepository,
    InMemoryProcessingResultRepository,
    InMemoryVisionAnalysisRepository,
)
from .services import (
    TesseractOCRService,
    WhisperTranscriptionService,
    OllamaCorrectionService,
    EventPublisherService,
)
from .llm_correction import (
    OllamaLLMCorrectionService,
    LLMCorrectionRequest,
    LLMCorrectionResult,
    get_llm_correction_service,
)
from .vision_analysis import (
    OllamaVisionAnalysisService,
    VisionAnalysisResult,
    get_vision_analysis_service,
)
from .output import (
    get_output_service,
    OutputFormat,
    ProcessingResultOutputService,
    OutputMetadata,
)

__all__ = [
    # Error Handling
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "ProcessingError",
    "ValidationError",
    "InfrastructureError",
    "DomainError",
    "ExternalServiceError",
    "SystemError",
    "ErrorHandler",
    "CircuitBreaker",
    # Logging
    "LogLevel",
    "LogEntry",
    "LoggingConfiguration",
    "CorrelationIdGenerator",
    "StructuredLogger",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "start_performance_tracking",
    "get_performance_metrics",
    "PerformanceMetrics",
    # Repository Implementations
    "InMemoryDocumentRepository",
    "InMemoryProcessingResultRepository",
    "InMemoryVisionAnalysisRepository",
    # Service Implementations
    "TesseractOCRService",
    "WhisperTranscriptionService",
    "OllamaCorrectionService",
    "EventPublisherService",
    # LLM Correction
    "OllamaLLMCorrectionService",
    "LLMCorrectionRequest",
    "LLMCorrectionResult",
    "get_llm_correction_service",
    # Vision Analysis
    "OllamaVisionAnalysisService",
    "VisionAnalysisResult",
    "get_vision_analysis_service",
    # Output Services
    "get_output_service",
    "OutputFormat",
    "ProcessingResultOutputService",
    "OutputMetadata",
]
