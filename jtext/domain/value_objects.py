"""
Domain Value Objects.

Value objects are immutable objects that represent descriptive aspects of the domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List
import uuid

from ..core import Result, Ok, Err, generate_id


@dataclass(frozen=True)
class DocumentId:
    """Document identifier value object."""

    value: str

    @classmethod
    def generate(cls) -> "DocumentId":
        """Generate a new document ID."""
        return cls(generate_id())

    def __post_init__(self) -> None:
        """Validate document ID."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Document ID must be a non-empty string")
        if len(self.value) > 255:
            raise ValueError("Document ID must be 255 characters or less")


class ProcessingStatus(Enum):
    """Document processing status."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentType(Enum):
    """Document type enumeration."""

    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    PDF = "PDF"


class ProcessingResultType(Enum):
    """Processing result type enumeration."""

    OCR = "OCR"
    TRANSCRIPTION = "TRANSCRIPTION"
    CORRECTION = "CORRECTION"
    VISION_ANALYSIS = "VISION_ANALYSIS"
    DOCX = "DOCX"
    UNKNOWN = "UNKNOWN"


class Language(Enum):
    """Language enumeration."""

    JAPANESE = "jpn"
    ENGLISH = "eng"
    MIXED = "jpn+eng"


@dataclass(frozen=True)
class FilePath:
    """File path value object."""

    value: str

    def __post_init__(self) -> None:
        """Validate file path."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("File path must be a non-empty string")
        if not Path(self.value).exists():
            raise ValueError(f"File does not exist: {self.value}")


@dataclass(frozen=True)
class Confidence:
    """Confidence value object."""

    value: Decimal

    @classmethod
    def from_float(cls, value: float) -> "Confidence":
        """Create confidence from float."""
        if not 0 <= value <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return cls(Decimal(str(value)))

    @classmethod
    def from_percentage(cls, percentage: float) -> "Confidence":
        """Create confidence from percentage."""
        return cls.from_float(percentage / 100.0)

    def to_float(self) -> float:
        """Convert to float."""
        return float(self.value)

    def to_percentage(self) -> float:
        """Convert to percentage."""
        return float(self.value) * 100.0


@dataclass(frozen=True)
class ProcessingMetrics:
    """Processing metrics value object."""

    processing_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    confidence_score: Confidence

    def __post_init__(self) -> None:
        """Validate metrics."""
        if self.processing_time_seconds < 0:
            raise ValueError("Processing time cannot be negative")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage cannot be negative")
        if not 0 <= self.cpu_usage_percent <= 100:
            raise ValueError("CPU usage must be between 0 and 100")


class VisionAnalysisType(Enum):
    """Vision analysis type enumeration."""

    COMPREHENSIVE = "COMPREHENSIVE"
    DOCUMENT_STRUCTURE = "DOCUMENT_STRUCTURE"
    TEXT_REGIONS = "TEXT_REGIONS"
    VISUAL_ELEMENTS = "VISUAL_ELEMENTS"
    QUALITY_ASSESSMENT = "QUALITY_ASSESSMENT"


@dataclass(frozen=True)
class VisionAnalysisId:
    """Vision analysis identifier value object."""

    value: str

    @classmethod
    def generate(cls) -> "VisionAnalysisId":
        """Generate a new vision analysis ID."""
        return cls(generate_id())

    def __post_init__(self) -> None:
        """Validate vision analysis ID."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Vision analysis ID must be a non-empty string")
        if len(self.value) > 255:
            raise ValueError("Vision analysis ID must be 255 characters or less")


@dataclass(frozen=True)
class TextRegion:
    """Text region value object."""

    area: str
    content_type: str
    confidence: Confidence
    bounding_box: Dict[str, float] = field(default_factory=dict)
    text_content: str = ""

    def __post_init__(self) -> None:
        """Validate text region."""
        if not self.area or not isinstance(self.area, str):
            raise ValueError("Area must be a non-empty string")
        if not self.content_type or not isinstance(self.content_type, str):
            raise ValueError("Content type must be a non-empty string")


@dataclass(frozen=True)
class VisualElement:
    """Visual element value object."""

    element_type: str
    description: str
    confidence: Confidence
    position: Dict[str, float] = field(default_factory=dict)
    size: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate visual element."""
        if not self.element_type or not isinstance(self.element_type, str):
            raise ValueError("Element type must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Description must be a non-empty string")


@dataclass(frozen=True)
class QualityAssessment:
    """Quality assessment value object."""

    clarity: float
    contrast: float
    readability: float
    overall_quality: float
    enhancement_suggestions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate quality assessment."""
        for score in [
            self.clarity,
            self.contrast,
            self.readability,
            self.overall_quality,
        ]:
            if not 0 <= score <= 1:
                raise ValueError("Quality scores must be between 0 and 1")


@dataclass(frozen=True)
class DocumentStructure:
    """Document structure value object."""

    document_type: str
    main_sections: List[str] = field(default_factory=list)
    layout_type: str = "unknown"
    reading_direction: str = "left-to-right"

    def __post_init__(self) -> None:
        """Validate document structure."""
        if not self.document_type or not isinstance(self.document_type, str):
            raise ValueError("Document type must be a non-empty string")
