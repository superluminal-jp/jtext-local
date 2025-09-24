"""
Domain Services.

Domain services contain business logic that doesn't naturally fit in entities or value objects.
"""

from pathlib import Path

from .entities import Document
from .value_objects import DocumentType, ProcessingStatus


class DocumentProcessingService:
    """Domain service for document processing logic."""

    @staticmethod
    def can_process_document(document: Document) -> bool:
        """Check if document can be processed."""
        return document.status == ProcessingStatus.PENDING

    @staticmethod
    def calculate_processing_priority(document: Document) -> int:
        """Calculate processing priority (higher number = higher priority)."""
        priority = 0

        # File size factor
        file_size_mb = Path(document.file_path.value).stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            priority += 3
        elif file_size_mb > 10:
            priority += 2
        else:
            priority += 1

        # Document type factor
        if document.document_type == DocumentType.IMAGE:
            priority += 2
        elif document.document_type == DocumentType.AUDIO:
            priority += 1

        return priority
