"""
Tests for domain layer components.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from jtext.domain import (
    DocumentId,
    ProcessingStatus,
    DocumentType,
    Language,
    FilePath,
    Confidence,
    ProcessingMetrics,
    DomainEvent,
    DocumentProcessedEvent,
    ProcessingFailedEvent,
    Document,
    ImageDocument,
    AudioDocument,
    ProcessingResult,
    OCRResult,
    TranscriptionResult,
    DocumentProcessingService,
)


class TestValueObjects:
    """Test value objects."""

    def test_document_id_generation(self):
        """Test document ID generation."""
        doc_id = DocumentId.generate()
        assert isinstance(doc_id, DocumentId)
        assert len(doc_id.value) > 0

    def test_document_id_validation(self):
        """Test document ID validation."""
        # Valid ID
        doc_id = DocumentId("valid-id")
        assert doc_id.value == "valid-id"

        # Invalid ID - empty
        with pytest.raises(ValueError):
            DocumentId("")

        # Invalid ID - too long
        with pytest.raises(ValueError):
            DocumentId("x" * 256)

    def test_confidence_from_float(self):
        """Test confidence creation from float."""
        conf = Confidence.from_float(0.85)
        assert conf.value == Decimal("0.85")

        # Invalid confidence
        with pytest.raises(ValueError):
            Confidence.from_float(1.5)

        with pytest.raises(ValueError):
            Confidence.from_float(-0.1)

    def test_confidence_from_percentage(self):
        """Test confidence creation from percentage."""
        conf = Confidence.from_percentage(85.0)
        assert conf.value == Decimal("0.85")

    def test_confidence_conversion(self):
        """Test confidence conversion methods."""
        conf = Confidence.from_float(0.75)
        assert conf.to_float() == 0.75
        assert conf.to_percentage() == 75.0


class TestDomainEvents:
    """Test domain events."""

    def test_domain_event_creation(self):
        """Test domain event creation."""
        event = DomainEvent(aggregate_id="test-123")
        assert event.aggregate_id == "test-123"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_document_processed_event(self):
        """Test document processed event."""
        event = DocumentProcessedEvent(
            document_id="doc-123",
            processing_result={"text": "extracted text"},
            processing_time=2.5,
        )
        assert event.document_id == "doc-123"
        assert event.processing_result["text"] == "extracted text"
        assert event.processing_time == 2.5

    def test_processing_failed_event(self):
        """Test processing failed event."""
        event = ProcessingFailedEvent(
            document_id="doc-123",
            error_message="Processing failed",
            error_code="PROCESSING_ERROR",
        )
        assert event.document_id == "doc-123"
        assert event.error_message == "Processing failed"
        assert event.error_code == "PROCESSING_ERROR"


class TestDomainEntities:
    """Test domain entities."""

    def test_document_creation(self, test_file_path):
        """Test document creation."""
        doc_id = DocumentId.generate()

        document = Document(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert document.id == doc_id
        assert document.file_path == test_file_path
        assert document.document_type == DocumentType.IMAGE
        assert document.status == ProcessingStatus.PENDING

    def test_document_status_update(self, test_file_path):
        """Test document status update."""
        doc_id = DocumentId.generate()

        document = Document(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        document.update_status(ProcessingStatus.PROCESSING)
        assert document.status == ProcessingStatus.PROCESSING
        assert document.updated_at > document.created_at

    def test_image_document_creation(self, test_file_path):
        """Test image document creation."""
        doc_id = DocumentId.generate()

        image_doc = ImageDocument(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            width=1920,
            height=1080,
            dpi=300,
        )

        assert image_doc.width == 1920
        assert image_doc.height == 1080
        assert image_doc.dpi == 300


class TestDomainServices:
    """Test domain services."""

    def test_can_process_document(self, test_file_path):
        """Test document processing capability check."""
        doc_id = DocumentId.generate()

        # Pending document can be processed
        pending_doc = Document(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert DocumentProcessingService.can_process_document(pending_doc)

        # Processing document cannot be processed again
        processing_doc = Document(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PROCESSING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert not DocumentProcessingService.can_process_document(processing_doc)

    def test_calculate_processing_priority(self, test_file_path):
        """Test processing priority calculation."""
        doc_id = DocumentId.generate()

        # Image document should have higher priority
        image_doc = Document(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        priority = DocumentProcessingService.calculate_processing_priority(image_doc)
        assert priority > 0
