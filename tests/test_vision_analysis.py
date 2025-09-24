"""
Tests for Vision Analysis functionality.

Tests the complete vision analysis pipeline including domain entities,
use cases, infrastructure services, and CLI integration.
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from pathlib import Path

from jtext.core import Result, Ok, Err
from jtext.domain import (
    VisionAnalysis,
    VisionAnalysisId,
    VisionAnalysisType,
    DocumentId,
    Language,
    Confidence,
    TextRegion,
    VisualElement,
    QualityAssessment,
    DocumentStructure,
)
from jtext.application import (
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    VisionAnalysisUseCase,
    GetVisionAnalysisUseCase,
    ListVisionAnalysesUseCase,
)
from jtext.infrastructure import (
    OllamaVisionAnalysisService,
    InMemoryVisionAnalysisRepository,
)


class TestVisionAnalysisDomain:
    """Test vision analysis domain entities and value objects."""

    def test_vision_analysis_creation(self):
        """Test vision analysis entity creation."""
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        assert analysis.id == analysis_id
        assert analysis.document_id == document_id
        assert analysis.analysis_type == VisionAnalysisType.COMPREHENSIVE
        assert analysis.confidence == confidence
        assert analysis.processing_time_ms == 1500.0
        assert analysis.model_used == "gemma3:4b"

    def test_vision_analysis_to_dict(self):
        """Test vision analysis to dictionary conversion."""
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        result = analysis.to_dict()

        assert result["id"] == analysis_id.value
        assert result["document_id"] == document_id.value
        assert result["analysis_type"] == "COMPREHENSIVE"
        assert result["confidence"] == 0.85
        assert result["processing_time_ms"] == 1500.0
        assert result["model_used"] == "gemma3:4b"

    def test_text_region_creation(self):
        """Test text region value object creation."""
        confidence = Confidence.from_float(0.9)

        region = TextRegion(
            area="header",
            content_type="title",
            confidence=confidence,
            bounding_box={"x": 0, "y": 0, "width": 100, "height": 20},
            text_content="Document Title",
        )

        assert region.area == "header"
        assert region.content_type == "title"
        assert region.confidence == confidence
        assert region.bounding_box == {"x": 0, "y": 0, "width": 100, "height": 20}
        assert region.text_content == "Document Title"

    def test_visual_element_creation(self):
        """Test visual element value object creation."""
        confidence = Confidence.from_float(0.8)

        element = VisualElement(
            element_type="image",
            description="Company logo",
            confidence=confidence,
            position={"x": 10, "y": 10},
            size={"width": 30, "height": 20},
        )

        assert element.element_type == "image"
        assert element.description == "Company logo"
        assert element.confidence == confidence
        assert element.position == {"x": 10, "y": 10}
        assert element.size == {"width": 30, "height": 20}

    def test_quality_assessment_creation(self):
        """Test quality assessment value object creation."""
        assessment = QualityAssessment(
            clarity=0.85,
            contrast=0.90,
            readability=0.88,
            overall_quality=0.87,
            enhancement_suggestions=["Improve contrast", "Enhance text sharpness"],
        )

        assert assessment.clarity == 0.85
        assert assessment.contrast == 0.90
        assert assessment.readability == 0.88
        assert assessment.overall_quality == 0.87
        assert assessment.enhancement_suggestions == [
            "Improve contrast",
            "Enhance text sharpness",
        ]

    def test_document_structure_creation(self):
        """Test document structure value object creation."""
        structure = DocumentStructure(
            document_type="business_letter",
            main_sections=["header", "body", "signature"],
            layout_type="single_column",
            reading_direction="left-to-right",
        )

        assert structure.document_type == "business_letter"
        assert structure.main_sections == ["header", "body", "signature"]
        assert structure.layout_type == "single_column"
        assert structure.reading_direction == "left-to-right"


class TestVisionAnalysisRepository:
    """Test vision analysis repository implementation."""

    def test_save_vision_analysis(self):
        """Test saving vision analysis."""
        repo = InMemoryVisionAnalysisRepository()
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        result = repo.save(analysis)
        assert result.is_ok

    def test_find_vision_analysis_by_id(self):
        """Test finding vision analysis by ID."""
        repo = InMemoryVisionAnalysisRepository()
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        repo.save(analysis)
        result = repo.find_by_id(analysis_id)

        assert result.is_ok
        found_analysis = result.unwrap()
        assert found_analysis.id == analysis_id

    def test_find_vision_analyses_by_document_id(self):
        """Test finding vision analyses by document ID."""
        repo = InMemoryVisionAnalysisRepository()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis1 = VisionAnalysis(
            id=VisionAnalysisId.generate(),
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        analysis2 = VisionAnalysis(
            id=VisionAnalysisId.generate(),
            document_id=document_id,
            analysis_type=VisionAnalysisType.DOCUMENT_STRUCTURE,
            confidence=confidence,
            processing_time_ms=1200.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        repo.save(analysis1)
        repo.save(analysis2)

        result = repo.find_by_document_id(document_id)
        assert result.is_ok

        analyses = result.unwrap()
        assert len(analyses) == 2

    def test_delete_vision_analysis(self):
        """Test deleting vision analysis."""
        repo = InMemoryVisionAnalysisRepository()
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        repo.save(analysis)

        # Verify it exists
        result = repo.find_by_id(analysis_id)
        assert result.is_ok

        # Delete it
        delete_result = repo.delete(analysis_id)
        assert delete_result.is_ok

        # Verify it's gone
        result = repo.find_by_id(analysis_id)
        assert result.is_ok
        assert result.unwrap() is None


class TestVisionAnalysisUseCase:
    """Test vision analysis use case."""

    def test_vision_analysis_use_case_execution(self):
        """Test vision analysis use case execution."""
        # Mock dependencies
        document_repo = Mock()
        vision_analysis_repo = InMemoryVisionAnalysisRepository()
        vision_analysis_service = Mock()
        event_publisher = Mock()

        # Mock vision analysis service
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        mock_analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        vision_analysis_service.analyze_document.return_value = Ok(mock_analysis)

        # Mock document repository
        document_repo.save.return_value = Ok(None)

        # Create use case
        use_case = VisionAnalysisUseCase(
            document_repo,
            vision_analysis_repo,
            vision_analysis_service,
            event_publisher,
        )

        # Create request
        request = VisionAnalysisRequest(
            image_path="test_image.jpg",
            language=Language.JAPANESE,
            analysis_type="comprehensive",
            model_name="gemma3:4b",
        )

        # Execute use case
        result = use_case.execute(request)

        assert result.is_ok
        response = result.unwrap()
        assert response.analysis_id == analysis_id.value
        assert response.document_id == document_id.value
        assert response.analysis_type == "COMPREHENSIVE"
        assert response.confidence == 0.85

    def test_vision_analysis_use_case_failure(self):
        """Test vision analysis use case failure."""
        # Mock dependencies
        document_repo = Mock()
        vision_analysis_repo = Mock()
        vision_analysis_service = Mock()
        event_publisher = Mock()

        # Mock vision analysis service to fail
        vision_analysis_service.analyze_document.return_value = Err("Analysis failed")

        # Create use case
        use_case = VisionAnalysisUseCase(
            document_repo,
            vision_analysis_repo,
            vision_analysis_service,
            event_publisher,
        )

        # Create request
        request = VisionAnalysisRequest(
            image_path="test_image.jpg",
            language=Language.JAPANESE,
            analysis_type="comprehensive",
        )

        # Execute use case
        result = use_case.execute(request)

        assert result.is_err
        assert "Vision analysis failed" in result.unwrap_err()


class TestOllamaVisionAnalysisService:
    """Test Ollama vision analysis service."""

    @patch("jtext.infrastructure.vision_analysis.requests")
    def test_vision_analysis_service_connection_test(self, mock_requests):
        """Test vision analysis service connection test."""
        # Mock successful connection
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "gemma3:4b"}]}
        mock_requests.get.return_value = mock_response

        service = OllamaVisionAnalysisService()
        assert service.is_available()

    @patch("jtext.infrastructure.vision_analysis.requests")
    def test_vision_analysis_service_connection_failure(self, mock_requests):
        """Test vision analysis service connection failure."""
        # Mock connection failure
        mock_requests.get.side_effect = Exception("Connection failed")

        service = OllamaVisionAnalysisService()
        assert not service.is_available()

    @patch("jtext.infrastructure.vision_analysis.requests")
    def test_vision_analysis_service_analyze_document(self, mock_requests):
        """Test vision analysis service document analysis."""
        # Mock successful API call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps(
                {
                    "document_structure": {
                        "document_type": "business_letter",
                        "main_sections": ["header", "body"],
                        "layout_type": "single_column",
                        "reading_direction": "left-to-right",
                    },
                    "text_regions": [
                        {
                            "area": "header",
                            "content_type": "title",
                            "confidence": 0.95,
                            "bounding_box": {
                                "x": 0,
                                "y": 0,
                                "width": 100,
                                "height": 20,
                            },
                            "text_content": "Document Title",
                        }
                    ],
                    "visual_elements": [
                        {
                            "element_type": "image",
                            "description": "Company logo",
                            "confidence": 0.92,
                            "position": {"x": 10, "y": 10},
                            "size": {"width": 30, "height": 20},
                        }
                    ],
                    "quality_assessment": {
                        "clarity": 0.85,
                        "contrast": 0.90,
                        "readability": 0.88,
                        "overall_quality": 0.87,
                        "enhancement_suggestions": ["Improve contrast"],
                    },
                    "content_categories": ["business", "formal"],
                    "analysis_confidence": 0.89,
                    "reasoning": "Comprehensive analysis completed",
                }
            )
        }
        mock_requests.post.return_value = mock_response

        service = OllamaVisionAnalysisService()

        # Create test image file
        test_image_path = "test_image.jpg"
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"fake_image_data"
            )

            result = service.analyze_document(
                image_path=test_image_path,
                language=Language.JAPANESE,
                analysis_type="comprehensive",
                model_name="gemma3:4b",
            )

        assert result.is_ok
        analysis = result.unwrap()
        assert analysis.analysis_type == VisionAnalysisType.COMPREHENSIVE
        assert analysis.confidence.to_float() == 0.89
        assert len(analysis.text_regions) == 1
        assert len(analysis.visual_elements) == 1
        assert analysis.quality_assessment is not None

    @patch("jtext.infrastructure.vision_analysis.requests")
    def test_vision_analysis_service_api_failure(self, mock_requests):
        """Test vision analysis service API failure."""
        # Mock API failure
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_requests.post.return_value = mock_response

        service = OllamaVisionAnalysisService()

        # Create test image file
        test_image_path = "test_image.jpg"
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"fake_image_data"
            )

            result = service.analyze_document(
                image_path=test_image_path,
                language=Language.JAPANESE,
                analysis_type="comprehensive",
            )

        assert result.is_err
        assert "Ollama API error" in result.unwrap_err()


class TestVisionAnalysisIntegration:
    """Test vision analysis integration scenarios."""

    def test_complete_vision_analysis_workflow(self):
        """Test complete vision analysis workflow."""
        # This would test the full integration from CLI to domain
        # For now, we'll test the key components work together

        # Test domain entities
        analysis_id = VisionAnalysisId.generate()
        document_id = DocumentId.generate()
        confidence = Confidence.from_float(0.85)

        analysis = VisionAnalysis(
            id=analysis_id,
            document_id=document_id,
            analysis_type=VisionAnalysisType.COMPREHENSIVE,
            confidence=confidence,
            processing_time_ms=1500.0,
            model_used="gemma3:4b",
            created_at=datetime.now(timezone.utc),
        )

        # Test repository
        repo = InMemoryVisionAnalysisRepository()
        save_result = repo.save(analysis)
        assert save_result.is_ok

        # Test retrieval
        find_result = repo.find_by_id(analysis_id)
        assert find_result.is_ok
        retrieved_analysis = find_result.unwrap()
        assert retrieved_analysis.id == analysis_id

    def test_vision_analysis_error_handling(self):
        """Test vision analysis error handling."""
        # Test invalid confidence
        with pytest.raises(ValueError):
            Confidence.from_float(1.5)  # Invalid confidence > 1.0

        # Test invalid quality assessment
        with pytest.raises(ValueError):
            QualityAssessment(
                clarity=1.5,  # Invalid clarity > 1.0
                contrast=0.5,
                readability=0.5,
                overall_quality=0.5,
            )


if __name__ == "__main__":
    pytest.main([__file__])
