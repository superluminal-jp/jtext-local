"""
Tests for multimodal OCR functionality.

This module contains comprehensive tests for the MultimodalOCR class
and its integration with vision models and LLM correction.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import base64

from jtext.core.multimodal_ocr import MultimodalOCR, MultimodalOCRResult
from jtext.utils.logging import setup_logging

# Setup logging for tests
setup_logging(verbose=True)


class TestMultimodalOCRResult:
    """Tests for MultimodalOCRResult class."""

    def test_multimodal_result_creation(self):
        """Test MultimodalOCRResult creation."""
        result = MultimodalOCRResult(
            source_path="test.png",
            text="テストテキスト",
            confidence=0.85,
            processing_time=1.0,
            memory_usage=10.0,
            corrections_applied=2,
            correction_ratio=0.1,
            vision_analysis={"model": "llava", "confidence": 0.9},
            fusion_method="vision_fusion"
        )

        assert result.text == "テストテキスト"
        assert result.confidence == 0.85
        assert result.corrections_applied == 2
        assert result.fusion_method == "vision_fusion"
        assert result.vision_analysis["model"] == "llava"

    def test_multimodal_result_to_dict(self):
        """Test MultimodalOCRResult to_dict conversion."""
        result = MultimodalOCRResult(
            source_path="/test/image.png",
            text="テストテキスト",
            confidence=0.85,
            processing_time=5.2,
            memory_usage=15.0,
            corrections_applied=3,
            correction_ratio=0.15,
            vision_analysis={
                "model": "llava",
                "analysis": "画像分析結果",
                "confidence": 0.9
            },
            fusion_method="vision_fusion"
        )

        data = result.to_dict()

        assert data["source"] == str(Path("/test/image.png").absolute())
        assert data["type"] == "image"
        assert data["processing"]["fusion_method"] == "vision_fusion"
        assert data["processing"]["pipeline"] == ["tesseract", "vision_analysis", "llm_correction"]
        assert data["vision_analysis"]["model"] == "llava"
        assert data["correction_stats"]["characters_changed"] == 3


class TestMultimodalOCR:
    """Tests for MultimodalOCR class."""

    def test_multimodal_ocr_initialization(self):
        """Test MultimodalOCR initialization."""
        ocr = MultimodalOCR(
            llm_model="llama2",
            vision_model="llava",
            enable_correction=True,
            enable_vision=True
        )

        assert ocr.enable_correction is True
        assert ocr.enable_vision is True
        assert ocr.llm_model == "llama2"
        assert ocr.vision_model == "llava"

    def test_get_supported_vision_models(self):
        """Test getting supported vision models."""
        ocr = MultimodalOCR()
        models = ocr.get_supported_vision_models()

        assert "llava" in models
        assert "llava:7b" in models
        assert "llava:13b" in models
        assert "bakllava" in models

    @patch("jtext.core.multimodal_ocr.validate_image_file", return_value=True)
    @patch("jtext.core.multimodal_ocr.Image.open")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_string")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_data")
    @patch("jtext.core.multimodal_ocr.ImagePreprocessor.preprocess")
    def test_process_image_ocr_only(self, mock_preprocess, mock_data, mock_string, mock_open, mock_validate):
        """Test image processing with OCR only."""
        ocr = MultimodalOCR(enable_correction=False, enable_vision=False)

        # Setup mocks
        mock_image = Mock()
        mock_open.return_value = mock_image
        mock_string.return_value = "テストテキスト"
        mock_data.return_value = {"conf": ["85", "90", "80"]}
        
        # Mock preprocessed image
        mock_processed_image = Mock()
        mock_preprocess.return_value = mock_processed_image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = ocr.process_image(tmp_path)

            assert result.text == "テストテキスト"
            assert result.fusion_method == "ocr_only"
            assert result.vision_analysis == {}

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.core.multimodal_ocr.validate_image_file", return_value=True)
    @patch("jtext.core.multimodal_ocr.Image.open")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_string")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_data")
    @patch("jtext.core.multimodal_ocr.requests.post")
    @patch("jtext.core.multimodal_ocr.ImagePreprocessor.preprocess")
    def test_process_image_with_vision(self, mock_preprocess, mock_post, mock_data, mock_string, mock_open, mock_validate):
        """Test image processing with vision analysis."""
        ocr = MultimodalOCR(enable_correction=False, enable_vision=True)

        # Setup mocks
        mock_image = Mock()
        mock_open.return_value = mock_image
        mock_string.return_value = "テストテキスト"
        mock_data.return_value = {"conf": ["85", "90"]}
        
        # Mock preprocessed image
        mock_processed_image = Mock()
        mock_preprocess.return_value = mock_processed_image

        # Mock vision analysis response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "画像には日本語のテキストが含まれています。技術文書のようです。"
        }
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = ocr.process_image(tmp_path)

            assert result.text == "テストテキスト"
            assert result.vision_analysis is not None
            assert result.vision_analysis["model"] == "llava"
            assert "技術文書" in result.vision_analysis["analysis"]

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.core.multimodal_ocr.validate_image_file", return_value=True)
    @patch("jtext.core.multimodal_ocr.Image.open")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_string")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_data")
    @patch("jtext.core.multimodal_ocr.requests.post")
    @patch("jtext.core.multimodal_ocr.ImagePreprocessor.preprocess")
    def test_process_image_with_correction(self, mock_preprocess, mock_post, mock_data, mock_string, mock_open, mock_validate):
        """Test image processing with LLM correction."""
        ocr = MultimodalOCR(enable_correction=True, enable_vision=False)

        # Setup mocks
        mock_image = Mock()
        mock_open.return_value = mock_image
        mock_string.return_value = "テストテキスト"
        mock_data.return_value = {"conf": ["85", "90"]}
        
        # Mock preprocessed image
        mock_processed_image = Mock()
        mock_preprocess.return_value = mock_processed_image

        # Mock correction
        with patch.object(ocr.corrector, "correct", return_value=("修正されたテキスト", 2)):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = ocr.process_image(tmp_path)

                assert result.text == "修正されたテキスト"
                assert result.corrections_applied == 2
                assert result.fusion_method == "ocr_correction"

            finally:
                Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.core.multimodal_ocr.validate_image_file", return_value=True)
    @patch("jtext.core.multimodal_ocr.Image.open")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_string")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_data")
    @patch("jtext.core.multimodal_ocr.requests.post")
    @patch("jtext.core.multimodal_ocr.ImagePreprocessor.preprocess")
    def test_process_image_full_multimodal(self, mock_preprocess, mock_post, mock_data, mock_string, mock_open, mock_validate):
        """Test full multimodal processing with vision and multimodal fusion correction."""
        ocr = MultimodalOCR(enable_correction=True, enable_vision=True)

        # Setup mocks
        mock_image = Mock()
        mock_open.return_value = mock_image
        mock_string.return_value = "テストテキスト"
        mock_data.return_value = {"conf": ["85", "90"]}
        
        # Mock preprocessed image
        mock_processed_image = Mock()
        mock_preprocess.return_value = mock_processed_image

        # Mock vision analysis response (first call)
        mock_vision_response = Mock()
        mock_vision_response.status_code = 200
        mock_vision_response.json.return_value = {
            "response": "画像には日本語のテキストが含まれています。技術文書のようです。表や図表も含まれています。"
        }

        # Mock multimodal fusion response (second call)
        mock_fusion_response = Mock()
        mock_fusion_response.status_code = 200
        mock_fusion_response.json.return_value = {
            "response": "修正された高精度テキスト"
        }

        # Setup mock to return different responses for different calls
        mock_post.side_effect = [mock_vision_response, mock_fusion_response]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = ocr.process_image(tmp_path)

            assert result.text == "修正された高精度テキスト"
            assert result.corrections_applied > 0
            assert result.fusion_method == "multimodal_fusion"
            assert result.vision_analysis is not None
            assert "技術文書" in result.vision_analysis["analysis"]

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.core.multimodal_ocr.requests.post")
    def test_multimodal_fusion_correction(self, mock_post):
        """Test multimodal fusion correction method."""
        ocr = MultimodalOCR(enable_correction=True, enable_vision=True)
        
        # Mock successful multimodal fusion response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "修正された高精度テキスト\nこれは技術文書の内容です。"
        }
        mock_post.return_value = mock_response
        
        vision_analysis = {
            "analysis": "技術文書の画像です。表や図表が含まれています。",
            "model": "llava"
        }
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            corrected_text, corrections_applied = ocr._multimodal_fusion_correction(
                "テストテキスト", vision_analysis, tmp_path
            )
            
            assert corrected_text == "修正された高精度テキスト\nこれは技術文書の内容です。"
            assert corrections_applied > 0
            assert mock_post.called
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.core.multimodal_ocr.requests.post")
    def test_multimodal_fusion_fallback(self, mock_post):
        """Test multimodal fusion fallback when API fails."""
        ocr = MultimodalOCR(enable_correction=True, enable_vision=True)
        
        # Mock API failure
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        vision_analysis = {
            "analysis": "技術文書の画像です。",
            "model": "llava"
        }
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Should raise exception due to API failure
            with pytest.raises(Exception):
                ocr._multimodal_fusion_correction("テストテキスト", vision_analysis, tmp_path)
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_multimodal_fusion_prompt(self):
        """Test multimodal fusion prompt creation."""
        ocr = MultimodalOCR()
        
        vision_analysis = {
            "analysis": "技術文書の画像です。表が含まれています。",
            "model": "llava"
        }
        
        prompt = ocr._create_multimodal_fusion_prompt("OCRテキスト", vision_analysis)
        
        assert "OCRテキスト" in prompt
        assert "技術文書の画像です" in prompt
        assert "修正方針" in prompt
        assert "出力要件" in prompt
        assert "修正版テキスト:" in prompt

    def test_clean_multimodal_response(self):
        """Test multimodal response cleaning."""
        ocr = MultimodalOCR()
        
        # Test with prefix removal
        response = "修正版テキスト:\nこれは修正されたテキストです。"
        cleaned = ocr._clean_multimodal_response(response, "元のテキスト")
        assert cleaned == "これは修正されたテキストです。"
        
        # Test with too short response
        short_response = "短い"
        cleaned = ocr._clean_multimodal_response(short_response, "これは長い元のテキストです")
        assert cleaned == "これは長い元のテキストです"  # Should return original

    def test_calculate_corrections(self):
        """Test correction calculation."""
        ocr = MultimodalOCR()
        
        # Test identical texts
        corrections = ocr._calculate_corrections("同じテキスト", "同じテキスト")
        assert corrections == 0
        
        # Test different texts
        corrections = ocr._calculate_corrections("元のテキスト", "修正されたテキスト")
        assert corrections > 0
        
        # Test empty texts
        corrections = ocr._calculate_corrections("", "")
        assert corrections == 0

    def test_context_aware_fusion_fallback(self):
        """Test context-aware fusion fallback."""
        ocr = MultimodalOCR(enable_correction=True, enable_vision=True)
        
        vision_analysis = {
            "analysis": "技術文書の画像です。",
            "model": "llava"
        }
        
        # Mock context-aware correction
        with patch.object(ocr.context_corrector, "correct_with_context", return_value=("修正されたテキスト", 5)):
            result = ocr._context_aware_fusion_fallback("OCRテキスト", vision_analysis)
            
            corrected_text, corrections_applied, correction_ratio, fusion_method = result
            
            assert corrected_text == "修正されたテキスト"
            assert corrections_applied == 5
            assert correction_ratio > 0
            assert fusion_method == "context_aware_fallback"

    def test_extract_document_type(self):
        """Test document type extraction from vision analysis."""
        ocr = MultimodalOCR()

        # Test technical document
        vision_analysis = {"analysis": "技術文書の画像です"}
        doc_type = ocr._extract_document_type(vision_analysis)
        assert doc_type == "technical"

        # Test academic document
        vision_analysis = {"analysis": "学術論文の画像です"}
        doc_type = ocr._extract_document_type(vision_analysis)
        assert doc_type == "academic"

        # Test business document
        vision_analysis = {"analysis": "ビジネス文書の画像です"}
        doc_type = ocr._extract_document_type(vision_analysis)
        assert doc_type == "business"

        # Test tabular document
        vision_analysis = {"analysis": "表が含まれています"}
        doc_type = ocr._extract_document_type(vision_analysis)
        assert doc_type == "tabular"

        # Test general document
        vision_analysis = {"analysis": "一般的な文書です"}
        doc_type = ocr._extract_document_type(vision_analysis)
        assert doc_type == "general"

    def test_extract_layout_info(self):
        """Test layout information extraction from vision analysis."""
        ocr = MultimodalOCR()

        # Test structured layout
        vision_analysis = {"analysis": "構造化された文書で、表とリストが含まれています"}
        layout_info = ocr._extract_layout_info(vision_analysis)

        assert layout_info["has_tables"] is True
        assert layout_info["has_lists"] is True
        assert layout_info["layout_type"] == "structured"

        # Test freeform layout
        vision_analysis = {"analysis": "自由形式の文書です"}
        layout_info = ocr._extract_layout_info(vision_analysis)

        assert layout_info["has_tables"] is False
        assert layout_info["has_lists"] is False
        assert layout_info["layout_type"] == "freeform"

    @patch("jtext.core.multimodal_ocr.requests.get")
    def test_check_vision_model_availability_success(self, mock_get):
        """Test successful vision model availability check."""
        ocr = MultimodalOCR(vision_model="llava")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llava"}, {"name": "llama2"}]
        }
        mock_get.return_value = mock_response

        result = ocr.check_vision_model_availability()
        assert result is True

    @patch("jtext.core.multimodal_ocr.requests.get")
    def test_check_vision_model_availability_failure(self, mock_get):
        """Test vision model availability check failure."""
        ocr = MultimodalOCR(vision_model="llava")

        mock_get.side_effect = Exception("Connection failed")

        result = ocr.check_vision_model_availability()
        assert result is False


class TestMultimodalIntegration:
    """Integration tests for multimodal OCR functionality."""

    @patch("jtext.core.multimodal_ocr.validate_image_file", return_value=True)
    @patch("jtext.core.multimodal_ocr.Image.open")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_string")
    @patch("jtext.core.multimodal_ocr.pytesseract.image_to_data")
    @patch("jtext.core.multimodal_ocr.requests.post")
    @patch("jtext.core.multimodal_ocr.ImagePreprocessor.preprocess")
    def test_multimodal_pipeline_integration(self, mock_preprocess, mock_post, mock_data, mock_string, mock_open, mock_validate):
        """Test complete multimodal pipeline integration."""
        ocr = MultimodalOCR(
            llm_model="llama2",
            vision_model="llava",
            enable_correction=True,
            enable_vision=True
        )

        # Setup mocks
        mock_image = Mock()
        mock_open.return_value = mock_image
        mock_string.return_value = "OCRで抽出されたテキスト"
        mock_data.return_value = {"conf": ["85", "90", "80"]}
        
        # Mock preprocessed image
        mock_processed_image = Mock()
        mock_preprocess.return_value = mock_processed_image

        # Mock vision analysis (first call)
        mock_vision_response = Mock()
        mock_vision_response.status_code = 200
        mock_vision_response.json.return_value = {
            "response": "技術文書の画像です。表や図表が含まれています。"
        }

        # Mock multimodal fusion (second call)
        mock_fusion_response = Mock()
        mock_fusion_response.status_code = 200
        mock_fusion_response.json.return_value = {
            "response": "修正された高精度テキスト"
        }

        # Setup mock to return different responses for different calls
        mock_post.side_effect = [mock_vision_response, mock_fusion_response]

        # Mock context-aware correction
        with patch.object(ocr.context_corrector, "correct_with_context", return_value=("修正されたテキスト", 5)):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = ocr.process_image(tmp_path)

                # Verify all components worked
                assert result.text == "修正された高精度テキスト"
                assert result.corrections_applied > 0
                assert result.fusion_method == "multimodal_fusion"
                assert result.vision_analysis is not None
                assert result.vision_analysis["model"] == "llava"

                # Verify pipeline
                pipeline = result.to_dict()["processing"]["pipeline"]
                assert "tesseract" in pipeline
                assert "vision_analysis" in pipeline
                assert "llm_correction" in pipeline

            finally:
                Path(tmp_path).unlink(missing_ok=True)
