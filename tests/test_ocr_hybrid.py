"""
Tests for the hybrid OCR processing module.

This module contains unit tests for the HybridOCR class and
ProcessingResult data structure.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from jtext.core.ocr_hybrid import HybridOCR, ProcessingResult


class TestProcessingResult:
    """Test cases for ProcessingResult class."""

    def test_processing_result_creation(self):
        """Test ProcessingResult object creation."""
        result = ProcessingResult(
            source_path="/test/image.png",
            text="テストテキスト",
            confidence=0.85,
            processing_time=5.2,
            memory_usage=128.5,
            corrections_applied=3,
            correction_ratio=0.02,
        )

        assert result.source_path == "/test/image.png"
        assert result.text == "テストテキスト"
        assert result.confidence == 0.85
        assert result.processing_time == 5.2
        assert result.memory_usage == 128.5
        assert result.corrections_applied == 3
        assert result.correction_ratio == 0.02
        assert result.timestamp > 0

    def test_processing_result_to_dict(self):
        """Test ProcessingResult to_dict method."""
        result = ProcessingResult(
            source_path="/test/image.png",
            text="テストテキスト",
            confidence=0.85,
            processing_time=5.2,
            memory_usage=128.5,
            corrections_applied=3,
            correction_ratio=0.02,
        )

        data = result.to_dict()

        assert data["source"] == str(Path("/test/image.png").absolute())
        assert data["type"] == "image"
        assert data["processing"]["pipeline"] == ["tesseract", "llm_correction"]
        assert data["processing"]["llm_model"] == "gpt-oss"
        assert data["correction_stats"]["characters_changed"] == 3
        assert (
            data["quality_metrics"]["character_count"] == 7
        )  # "テストテキスト" = 7 characters
        assert data["quality_metrics"]["processing_time_sec"] == 5.2


class TestHybridOCR:
    """Test cases for HybridOCR class."""

    def test_hybrid_ocr_initialization(self):
        """Test HybridOCR initialization."""
        # Test without LLM correction
        ocr = HybridOCR(enable_correction=False)
        assert ocr.enable_correction is False
        assert ocr.corrector is None

        # Test with LLM correction
        ocr = HybridOCR(llm_model="gpt-oss", enable_correction=True)
        assert ocr.enable_correction is True
        assert ocr.llm_model == "gpt-oss"
        assert ocr.corrector is not None

    @patch("jtext.core.ocr_hybrid.pytesseract.image_to_string")
    @patch("jtext.core.ocr_hybrid.pytesseract.image_to_data")
    @patch("jtext.core.ocr_hybrid.Image.open")
    @patch("jtext.core.ocr_hybrid.validate_image_file")
    def test_process_image_success(
        self, mock_validate, mock_open, mock_data, mock_string
    ):
        """Test successful image processing."""
        # Setup mocks
        mock_validate.return_value = True
        mock_image = Mock(spec=Image.Image)
        mock_open.return_value = mock_image
        mock_string.return_value = "テストテキスト"
        mock_data.return_value = {"conf": ["85", "90", "80", "88"]}

        # Mock preprocessor and image processing
        with patch.object(HybridOCR, "_calculate_confidence", return_value=0.85):
            with patch(
                "jtext.preprocessing.image_prep.ImagePreprocessor.preprocess"
            ) as mock_preprocess:
                # Create a mock PIL Image that works with np.array()
                mock_image = Mock()
                mock_image.__array_interface__ = {
                    "shape": (100, 100, 3),
                    "typestr": "|u1",
                    "data": (id(mock_image), False),
                }
                mock_preprocess.return_value = mock_image

                ocr = HybridOCR(enable_correction=False)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    result = ocr.process_image(tmp_path)

                    assert result.text == "テストテキスト"
                    assert result.confidence == 0.85
                    assert result.source_path == tmp_path
                    assert result.processing_time > 0

                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    def test_process_image_invalid_file(self):
        """Test processing with invalid file."""
        ocr = HybridOCR()

        with pytest.raises(ValueError, match="Invalid image file"):
            ocr.process_image("/nonexistent/file.png")

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        ocr = HybridOCR()

        # Test with valid confidence data
        ocr_data = {"conf": ["85", "90", "80", "88", "0", "92"]}  # 0 should be ignored
        confidence = ocr._calculate_confidence(ocr_data)
        expected = (85 + 90 + 80 + 88 + 92) / 5 / 100.0
        assert confidence == pytest.approx(expected)

        # Test with no valid confidence data
        ocr_data = {"conf": ["0", "-1", "0"]}
        confidence = ocr._calculate_confidence(ocr_data)
        assert confidence == 0.0

    @patch("jtext.core.ocr_hybrid.pytesseract.image_to_string")
    @patch("jtext.core.ocr_hybrid.pytesseract.image_to_data")
    @patch("jtext.core.ocr_hybrid.Image.open")
    @patch("jtext.core.ocr_hybrid.validate_image_file")
    def test_process_image_with_correction(
        self, mock_validate, mock_open, mock_data, mock_string
    ):
        """Test image processing with LLM correction."""
        # Setup mocks
        mock_validate.return_value = True
        mock_image = Mock(spec=Image.Image)
        mock_open.return_value = mock_image
        mock_string.return_value = "テストテキスト"
        mock_data.return_value = {"conf": ["85", "90"]}

        # Mock corrector
        mock_corrector = Mock()
        mock_corrector.correct.return_value = ("修正されたテキスト", 2)

        with patch.object(HybridOCR, "_calculate_confidence", return_value=0.85):
            with patch(
                "jtext.preprocessing.image_prep.ImagePreprocessor.preprocess"
            ) as mock_preprocess:
                # Create a mock PIL Image that works with np.array()
                mock_processed_image = Mock()
                mock_processed_image.__array_interface__ = {
                    "shape": (100, 100, 3),
                    "typestr": "|u1",
                    "data": (id(mock_processed_image), False),
                }
                mock_preprocess.return_value = mock_processed_image

                ocr = HybridOCR(enable_correction=True)
                ocr.corrector = mock_corrector

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    result = ocr.process_image(tmp_path)

                    assert result.text == "修正されたテキスト"
                    assert result.corrections_applied == 2
                    assert result.correction_ratio > 0

                    # Verify corrector was called
                    mock_corrector.correct.assert_called_once_with("テストテキスト")

                finally:
                    Path(tmp_path).unlink(missing_ok=True)
