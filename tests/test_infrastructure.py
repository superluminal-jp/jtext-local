"""
Tests for infrastructure layer components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from jtext.infrastructure import (
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
    LogLevel,
    LogEntry,
    LoggingConfiguration,
    CorrelationIdGenerator,
    StructuredLogger,
    set_correlation_id,
    get_correlation_id,
    InMemoryDocumentRepository,
    InMemoryProcessingResultRepository,
    TesseractOCRService,
    WhisperTranscriptionService,
    OllamaCorrectionService,
    EventPublisherService,
)
from jtext.domain import (
    Document,
    ImageDocument,
    AudioDocument,
    ProcessingResult,
    DocumentId,
    DocumentType,
    Language,
    ProcessingStatus,
    Confidence,
    FilePath,
)


class TestErrorHandling:
    """Test error handling components."""

    def test_processing_error_creation(self):
        """Test processing error creation."""
        error = ProcessingError(
            message="Test error",
            error_code="TEST_ERROR",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DOMAIN,
        )

        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.DOMAIN
        assert error.timestamp is not None

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("Invalid input", field="email")
        assert error.message == "Invalid input"
        assert error.field == "email"
        assert error.severity == ErrorSeverity.LOW
        assert error.category == ErrorCategory.VALIDATION

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        breaker = CircuitBreaker()
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


class TestRepositories:
    """Test repository implementations."""

    def test_in_memory_document_repository(self, test_file_path):
        """Test in-memory document repository."""
        repo = InMemoryDocumentRepository()

        # Create test document
        doc_id = DocumentId.generate()
        document = ImageDocument(
            id=doc_id,
            file_path=test_file_path,
            document_type=DocumentType.IMAGE,
            language=Language.JAPANESE,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Test save
        result = repo.save(document)
        assert result.is_ok

        # Test find by ID
        result = repo.find_by_id(doc_id)
        assert result.is_ok
        found_doc = result.unwrap()
        assert found_doc.id == doc_id

        # Test find all
        result = repo.find_all()
        assert result.is_ok
        documents = result.unwrap()
        assert len(documents) == 1

        # Test delete
        result = repo.delete(doc_id)
        assert result.is_ok

        # Verify deletion
        result = repo.find_by_id(doc_id)
        assert result.is_ok
        assert result.unwrap() is None


class TestServices:
    """Test service implementations."""

    def test_event_publisher_service(self):
        """Test event publisher service."""
        publisher = EventPublisherService()

        # Test event publishing
        event = Mock()
        event.__class__.__name__ = "TestEvent"

        result = publisher.publish(event)
        assert result.is_ok

    @patch("jtext.infrastructure.services.pytesseract.image_to_string")
    @patch("jtext.infrastructure.services.pytesseract.image_to_data")
    @patch("jtext.infrastructure.services.Image.open")
    def test_tesseract_ocr_service(
        self, mock_image_open, mock_image_to_data, mock_image_to_string
    ):
        """Test Tesseract OCR service."""
        # Mock Tesseract response
        mock_image_to_string.return_value = "extracted text"
        mock_image_to_data.return_value = {"conf": [95, 90, 85]}

        # Mock Image.open to return a valid image object
        mock_image = Mock()
        mock_image_open.return_value = mock_image

        # Skip the service creation if Tesseract dependencies are not available
        try:
            service = TesseractOCRService()
        except ImportError:
            pytest.skip("Tesseract dependencies not available")

        # Test OCR extraction
        result = service.extract_text("test.jpg", Language.JAPANESE)
        assert result.is_ok

        ocr_result = result.unwrap()
        assert ocr_result.result_type == "OCR"
        assert ocr_result.content == "extracted text"
