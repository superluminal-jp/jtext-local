"""
Domain Events.

Domain events represent something important that happened in the domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import uuid


@dataclass
class DomainEvent:
    """Base domain event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    aggregate_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "event_type": self.__class__.__name__,
        }


@dataclass
class DocumentProcessedEvent(DomainEvent):
    """Event fired when document processing is completed."""

    document_id: str = ""
    processing_result: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class ProcessingFailedEvent(DomainEvent):
    """Event fired when document processing fails."""

    document_id: str = ""
    error_message: str = ""
    error_code: str = ""


@dataclass
class VisionAnalysisCompletedEvent(DomainEvent):
    """Event fired when vision analysis is completed."""

    analysis_id: str = ""
    document_id: str = ""
    analysis_type: str = ""
    confidence: float = 0.0
    processing_time_ms: float = 0.0
