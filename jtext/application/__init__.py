"""
Application layer - Use cases and DTOs.

This module contains application-specific logic including use cases,
data transfer objects, and application services following Clean Architecture.
"""

from .dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    DocumentDTO,
    ProcessingResultDTO,
    ProcessingStatisticsDTO,
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    VisionAnalysisDTO,
)
from .use_cases import (
    UseCase,
    ProcessDocumentUseCase,
    ProcessImageUseCase,
    ProcessAudioUseCase,
    ProcessStructuredDocumentUseCase,
    GetProcessingResultUseCase,
    ListDocumentsUseCase,
    GetProcessingStatisticsUseCase,
    VisionAnalysisUseCase,
    GetVisionAnalysisUseCase,
    ListVisionAnalysesUseCase,
)

__all__ = [
    # DTOs
    "ProcessDocumentRequest",
    "ProcessDocumentResponse",
    "DocumentDTO",
    "ProcessingResultDTO",
    "ProcessingStatisticsDTO",
    "VisionAnalysisRequest",
    "VisionAnalysisResponse",
    "VisionAnalysisDTO",
    # Use Cases
    "UseCase",
    "ProcessDocumentUseCase",
    "ProcessImageUseCase",
    "ProcessAudioUseCase",
    "ProcessStructuredDocumentUseCase",
    "GetProcessingResultUseCase",
    "ListDocumentsUseCase",
    "GetProcessingStatisticsUseCase",
    "VisionAnalysisUseCase",
    "GetVisionAnalysisUseCase",
    "ListVisionAnalysesUseCase",
]
