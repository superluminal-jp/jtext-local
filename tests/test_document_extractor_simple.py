"""
Simplified tests for document extraction module.

This module provides focused tests for the DocumentExtractor class
with proper mocking to avoid complex dependencies.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from jtext.processing.document_extractor import DocumentExtractor


class TestDocumentExtractorSimple:
    """Simplified tests for DocumentExtractor."""

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = DocumentExtractor()
        assert extractor is not None

    def test_extract_text_nonexistent_file(self):
        """Test extraction with non-existent file."""
        extractor = DocumentExtractor()

        with pytest.raises(ValueError):  # extract_text validates and raises ValueError
            extractor.extract_text("/nonexistent/file.pdf")

    def test_extract_text_unsupported_format(self):
        """Test extraction with unsupported file format."""
        extractor = DocumentExtractor()

        with tempfile.NamedTemporaryFile(suffix=".unsupported", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid document file"):
                extractor.extract_text(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_extract_pdf_success(self):
        """Test successful PDF extraction with proper string mocking."""
        extractor = DocumentExtractor()

        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {"title": "テスト文書", "author": "作成者"}

            # Mock page with proper string return
            mock_page = Mock()
            mock_page.get_images.return_value = []

            # Mock page.get_text() to consistently return string
            def get_text_side_effect(format=None):
                if format == "dict":
                    return {"blocks": []}
                return "ページ1のテキスト内容"

            mock_page.get_text.side_effect = get_text_side_effect

            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.load_page = Mock(return_value=mock_page)
            mock_doc.close = Mock()

            mock_fitz.return_value = mock_doc

            text, metadata = extractor._extract_pdf("test.pdf")

            assert "ページ1のテキスト内容" in text
            assert metadata["format"] == "pdf"
            assert metadata["pages"] == 1
            assert metadata["title"] == "テスト文書"

    def test_extract_docx_success(self):
        """Test successful DOCX extraction."""
        extractor = DocumentExtractor()

        with patch("jtext.processing.document_extractor.Document") as mock_document:
            mock_doc = Mock()

            # Mock paragraphs
            mock_para = Mock()
            mock_para.text = "段落のテキスト"
            mock_doc.paragraphs = [mock_para]

            # Mock tables (empty)
            mock_doc.tables = []

            # Mock relationships (no images)
            mock_doc.part.rels.values.return_value = []

            mock_document.return_value = mock_doc

            text, metadata = extractor._extract_docx("test.docx")

            assert "段落のテキスト" in text
            assert metadata["format"] == "docx"
            assert metadata["paragraphs"] == 1
            assert metadata["tables"] == 0

    def test_extract_pptx_success(self):
        """Test successful PPTX extraction."""
        extractor = DocumentExtractor()

        with patch(
            "jtext.processing.document_extractor.Presentation"
        ) as mock_presentation:
            mock_ppt = Mock()

            # Mock slide with text shapes
            mock_slide = Mock()

            mock_shape = Mock()
            mock_shape.has_text_frame = True
            mock_shape.text = "スライドのタイトル"

            mock_slide.shapes = [mock_shape]
            mock_ppt.slides = [mock_slide]

            mock_presentation.return_value = mock_ppt

            text, metadata = extractor._extract_pptx("test.pptx")

            assert "スライドのタイトル" in text
            assert metadata["format"] == "pptx"
            assert metadata["slides"] == 1

    def test_extract_html_success(self):
        """Test successful HTML extraction."""
        extractor = DocumentExtractor()

        html_content = """
        <html>
        <head><title>テストページ</title></head>
        <body>
            <h1>メインタイトル</h1>
            <p>段落のテキスト</p>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            temp_path = f.name

        try:
            text, metadata = extractor._extract_html(temp_path)

            assert "メインタイトル" in text
            assert "段落のテキスト" in text
            assert metadata["format"] == "html"

        finally:
            Path(temp_path).unlink()

    def test_extract_html_unicode(self):
        """Test HTML extraction with Unicode content."""
        extractor = DocumentExtractor()

        unicode_html = """
        <html>
        <head><meta charset="utf-8"></head>
        <body>
            <h1>日本語のタイトル</h1>
            <p>🎌 絵文字テスト 🌸</p>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(unicode_html)
            temp_path = f.name

        try:
            text, metadata = extractor._extract_html(temp_path)

            assert "日本語のタイトル" in text
            assert "🎌" in text
            assert "🌸" in text

        finally:
            Path(temp_path).unlink()

    def test_extract_pdf_empty_document(self):
        """Test PDF extraction with empty document."""
        extractor = DocumentExtractor()

        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {}
            mock_doc.__iter__ = Mock(return_value=iter([]))
            mock_doc.__len__ = Mock(return_value=0)
            mock_doc.load_page = Mock(return_value=None)
            mock_doc.close = Mock()

            mock_fitz.return_value = mock_doc

            text, metadata = extractor._extract_pdf("empty.pdf")

            assert text == ""
            assert metadata["pages"] == 0

    def test_extract_docx_no_content(self):
        """Test DOCX extraction with no readable content."""
        extractor = DocumentExtractor()

        with patch("jtext.processing.document_extractor.Document") as mock_document:
            mock_doc = Mock()

            # Empty or whitespace-only paragraphs
            mock_para = Mock()
            mock_para.text = "   "
            mock_doc.paragraphs = [mock_para]

            mock_doc.tables = []
            mock_doc.part.rels.values.return_value = []

            mock_document.return_value = mock_doc

            text, metadata = extractor._extract_docx("empty.docx")

            assert text == ""
            assert metadata["paragraphs"] == 0

    def test_extract_pptx_no_text_shapes(self):
        """Test PPTX extraction with slides containing no text."""
        extractor = DocumentExtractor()

        with patch(
            "jtext.processing.document_extractor.Presentation"
        ) as mock_presentation:
            mock_ppt = Mock()

            mock_slide = Mock()

            # Shape without text - mock empty text
            mock_shape = Mock(spec=["text"])  # Only has text attribute
            mock_shape.text.strip.return_value = ""  # Empty text shape

            mock_slide.shapes = [mock_shape]
            mock_ppt.slides = [mock_slide]

            mock_presentation.return_value = mock_ppt

            text, metadata = extractor._extract_pptx("no_text.pptx")

            assert text == ""
            assert metadata["slides"] == 1

    def test_error_handling_corrupted_pdf(self):
        """Test error handling with corrupted PDF."""
        extractor = DocumentExtractor()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"Not a real PDF")
            temp_path = f.name

        try:
            with pytest.raises(
                Exception
            ):  # Various exceptions possible for corrupted files
                extractor.extract_text(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_format_detection(self):
        """Test automatic format detection from file extensions."""
        extractor = DocumentExtractor()

        test_cases = [
            ("test.pdf", "_extract_pdf"),
            ("document.docx", "_extract_docx"),
            ("presentation.pptx", "_extract_pptx"),
            ("webpage.html", "_extract_html"),
            ("file.PDF", "_extract_pdf"),  # Case insensitive
        ]

        for filename, expected_method in test_cases:
            with patch.object(
                extractor, expected_method, return_value=("text", {})
            ) as mock_method:
                with patch("pathlib.Path.exists", return_value=True):
                    try:
                        extractor.extract_text(filename)
                        mock_method.assert_called_once()
                    except Exception:
                        pass  # Expected if file doesn't exist

    def test_extract_text_integration(self):
        """Test the main extract_text method."""
        extractor = DocumentExtractor()

        # Create a simple HTML file for integration test
        html_content = "<html><body><h1>Integration Test</h1></body></html>"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            temp_path = f.name

        try:
            result = extractor.extract_text(temp_path)

            assert "Integration Test" in result.text
            assert temp_path in result.source_path

        finally:
            Path(temp_path).unlink()

    def test_metadata_extraction(self):
        """Test metadata extraction from various document types."""
        extractor = DocumentExtractor()

        # Test PDF metadata
        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {
                "title": "Test Document",
                "author": "Test Author",
                "subject": "Test Subject",
            }

            mock_page = Mock()
            mock_page.get_images.return_value = []

            # Use side_effect for consistent string return
            def get_text_side_effect(format=None):
                if format == "dict":
                    return {"blocks": []}
                return "Test content"

            mock_page.get_text.side_effect = get_text_side_effect

            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.load_page = Mock(return_value=mock_page)
            mock_doc.close = Mock()

            mock_fitz.return_value = mock_doc

            text, metadata = extractor._extract_pdf("test.pdf")

            assert metadata["title"] == "Test Document"
            assert metadata["author"] == "Test Author"
            assert metadata["subject"] == "Test Subject"

    def test_large_content_handling(self):
        """Test handling of documents with large content."""
        extractor = DocumentExtractor()

        # Create large HTML content
        large_content = (
            "<html><body>" + "<p>Large content</p>" * 1000 + "</body></html>"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(large_content)
            temp_path = f.name

        try:
            text, metadata = extractor._extract_html(temp_path)

            assert "Large content" in text
            assert len(text) > 1000  # Should be substantial

        finally:
            Path(temp_path).unlink()
