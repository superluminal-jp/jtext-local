"""
Domain Entities.

Entities are objects with identity and lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .value_objects import (
    DocumentId,
    FilePath,
    DocumentType,
    Language,
    ProcessingStatus,
    Confidence,
    VisionAnalysisId,
    VisionAnalysisType,
    TextRegion,
    VisualElement,
    QualityAssessment,
    DocumentStructure,
)


@dataclass
class Document:
    """Base document entity."""

    id: DocumentId
    file_path: FilePath
    document_type: DocumentType
    language: Language
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, status: ProcessingStatus) -> None:
        """Update document status."""
        self.status = status
        self.updated_at = datetime.now()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "file_path": self.file_path.value,
            "document_type": self.document_type.value,
            "language": self.language.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ImageDocument(Document):
    """Image document entity."""

    width: Optional[int] = None
    height: Optional[int] = None
    dpi: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate image document."""
        if self.document_type != DocumentType.IMAGE:
            raise ValueError("ImageDocument must have IMAGE document type")


@dataclass
class AudioDocument(Document):
    """Audio document entity."""

    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate audio document."""
        if self.document_type != DocumentType.AUDIO:
            raise ValueError("AudioDocument must have AUDIO document type")


@dataclass
class ProcessingResult:
    """Processing result entity."""

    id: str
    document_id: DocumentId
    result_type: str
    content: str
    confidence: Confidence
    processing_time: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id.value,
            "result_type": self.result_type,
            "content": self.content,
            "confidence": self.confidence.to_float(),
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class OCRResult(ProcessingResult):
    """OCR processing result."""

    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    text_regions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate OCR result."""
        if self.result_type != "OCR":
            raise ValueError("OCRResult must have OCR result type")


@dataclass
class TranscriptionResult(ProcessingResult):
    """Transcription processing result."""

    segments: List[Dict[str, Any]] = field(default_factory=list)
    language_detected: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate transcription result."""
        if self.result_type != "TRANSCRIPTION":
            raise ValueError("TranscriptionResult must have TRANSCRIPTION result type")


@dataclass
class VisionAnalysis:
    """Vision analysis entity."""

    id: VisionAnalysisId
    document_id: DocumentId
    analysis_type: VisionAnalysisType
    confidence: Confidence
    processing_time_ms: float
    model_used: str
    created_at: datetime
    layout_structure: Optional[DocumentStructure] = None
    text_regions: List[TextRegion] = field(default_factory=list)
    visual_elements: List[VisualElement] = field(default_factory=list)
    quality_assessment: Optional[QualityAssessment] = None
    content_categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "document_id": self.document_id.value,
            "analysis_type": self.analysis_type.value,
            "confidence": self.confidence.to_float(),
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat(),
            "layout_structure": (
                {
                    "document_type": self.layout_structure.document_type,
                    "main_sections": self.layout_structure.main_sections,
                    "layout_type": self.layout_structure.layout_type,
                    "reading_direction": self.layout_structure.reading_direction,
                }
                if self.layout_structure
                else None
            ),
            "text_regions": [
                {
                    "area": region.area,
                    "content_type": region.content_type,
                    "confidence": region.confidence.to_float(),
                    "bounding_box": region.bounding_box,
                    "text_content": region.text_content,
                }
                for region in self.text_regions
            ],
            "visual_elements": [
                {
                    "element_type": element.element_type,
                    "description": element.description,
                    "confidence": element.confidence.to_float(),
                    "position": element.position,
                    "size": element.size,
                }
                for element in self.visual_elements
            ],
            "quality_assessment": (
                {
                    "clarity": self.quality_assessment.clarity,
                    "contrast": self.quality_assessment.contrast,
                    "readability": self.quality_assessment.readability,
                    "overall_quality": self.quality_assessment.overall_quality,
                    "enhancement_suggestions": self.quality_assessment.enhancement_suggestions,
                }
                if self.quality_assessment
                else None
            ),
            "content_categories": self.content_categories,
            "metadata": self.metadata,
        }

    def add_text_region(self, region: TextRegion) -> None:
        """Add text region to analysis."""
        self.text_regions.append(region)

    def add_visual_element(self, element: VisualElement) -> None:
        """Add visual element to analysis."""
        self.visual_elements.append(element)

    def set_quality_assessment(self, assessment: QualityAssessment) -> None:
        """Set quality assessment."""
        self.quality_assessment = assessment

    def set_layout_structure(self, structure: DocumentStructure) -> None:
        """Set layout structure."""
        self.layout_structure = structure

    def add_content_category(self, category: str) -> None:
        """Add content category."""
        if category not in self.content_categories:
            self.content_categories.append(category)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self.metadata[key] = value
