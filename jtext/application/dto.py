"""
Data Transfer Objects (DTOs).

DTOs define the shape of data crossing application boundaries.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ..domain import DocumentType, Language, ProcessingStatus


@dataclass
class ProcessDocumentRequest:
    """Request for document processing."""

    file_path: str
    document_type: DocumentType
    language: Language = Language.MIXED
    enable_correction: bool = False
    enable_vision: bool = False
    llm_model: Optional[str] = None
    vision_model: Optional[str] = None
    asr_model: Optional[str] = None
    output_format: str = "json"


@dataclass
class ProcessDocumentResponse:
    """Response for document processing."""

    document_id: str
    status: ProcessingStatus
    processing_time: float
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class DocumentDTO:
    """Document data transfer object."""

    id: str
    file_path: str
    document_type: str
    language: str
    status: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingResultDTO:
    """Processing result data transfer object."""

    id: str
    document_id: str
    result_type: str
    content: str
    confidence: float
    processing_time: float
    created_at: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingStatisticsDTO:
    """Processing statistics data transfer object."""

    total_documents: int
    completed_documents: int
    failed_documents: int
    average_processing_time: float
    success_rate: float
    documents_by_type: Dict[str, int]
    documents_by_status: Dict[str, int]


@dataclass
class VisionAnalysisRequest:
    """Request for vision analysis."""

    image_path: str
    language: Language
    analysis_type: str = "comprehensive"
    model_name: Optional[str] = None
    enable_quality_assessment: bool = True
    enable_structure_analysis: bool = True
    enable_visual_elements: bool = True


@dataclass
class VisionAnalysisResponse:
    """Response for vision analysis."""

    analysis_id: str
    document_id: str
    analysis_type: str
    confidence: float
    processing_time_ms: float
    model_used: str
    layout_structure: Optional[Dict[str, Any]] = None
    text_regions: List[Dict[str, Any]] = None
    visual_elements: List[Dict[str, Any]] = None
    quality_assessment: Optional[Dict[str, Any]] = None
    content_categories: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.text_regions is None:
            self.text_regions = []
        if self.visual_elements is None:
            self.visual_elements = []
        if self.content_categories is None:
            self.content_categories = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VisionAnalysisDTO:
    """Vision analysis data transfer object."""

    id: str
    document_id: str
    analysis_type: str
    confidence: float
    processing_time_ms: float
    model_used: str
    created_at: str
    layout_structure: Optional[Dict[str, Any]] = None
    text_regions: List[Dict[str, Any]] = None
    visual_elements: List[Dict[str, Any]] = None
    quality_assessment: Optional[Dict[str, Any]] = None
    content_categories: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.text_regions is None:
            self.text_regions = []
        if self.visual_elements is None:
            self.visual_elements = []
        if self.content_categories is None:
            self.content_categories = []
        if self.metadata is None:
            self.metadata = {}
