"""
Domain layer - Core business logic and entities.

This module contains the essential domain concepts following DDD principles.
"""

from .value_objects import (
    DocumentId,
    ProcessingStatus,
    DocumentType,
    ProcessingResultType,
    Language,
    FilePath,
    Confidence,
    ProcessingMetrics,
    VisionAnalysisId,
    VisionAnalysisType,
    TextRegion,
    VisualElement,
    QualityAssessment,
    DocumentStructure,
)
from .entities import (
    Document,
    ImageDocument,
    AudioDocument,
    ProcessingResult,
    OCRResult,
    TranscriptionResult,
    VisionAnalysis,
)
from .events import (
    DomainEvent,
    DocumentProcessedEvent,
    ProcessingFailedEvent,
    VisionAnalysisCompletedEvent,
)
from .services import DocumentProcessingService
from .repositories import (
    DocumentRepository,
    ProcessingResultRepository,
    VisionAnalysisRepository,
)
from .interfaces import (
    OCRService,
    TranscriptionService,
    CorrectionService,
    EventPublisher,
)

__all__ = [
    # Value Objects
    "DocumentId",
    "ProcessingStatus",
    "DocumentType",
    "ProcessingResultType",
    "Language",
    "FilePath",
    "Confidence",
    "ProcessingMetrics",
    "VisionAnalysisId",
    "VisionAnalysisType",
    "TextRegion",
    "VisualElement",
    "QualityAssessment",
    "DocumentStructure",
    # Domain Events
    "DomainEvent",
    "DocumentProcessedEvent",
    "ProcessingFailedEvent",
    "VisionAnalysisCompletedEvent",
    # Domain Entities
    "Document",
    "ImageDocument",
    "AudioDocument",
    "ProcessingResult",
    "OCRResult",
    "TranscriptionResult",
    "VisionAnalysis",
    # Domain Services
    "DocumentProcessingService",
    # Repository Interfaces
    "DocumentRepository",
    "ProcessingResultRepository",
    "VisionAnalysisRepository",
    "OCRService",
    "TranscriptionService",
    "CorrectionService",
    "EventPublisher",
]
