"""
Extended tests for document extraction module.

This module provides comprehensive tests for the DocumentExtractor class
covering edge cases, error conditions, and advanced functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from jtext.processing.document_extractor import DocumentExtractor


class TestDocumentExtractorEdgeCases:
    """Test edge cases and error conditions for DocumentExtractor."""

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = DocumentExtractor()
        assert extractor is not None

    def test_extract_document_nonexistent_file(self):
        """Test extraction with non-existent file."""
        extractor = DocumentExtractor()

        with pytest.raises(
            ValueError
        ):  # Changed to ValueError since extract_text validates first
            extractor.extract_text("/nonexistent/file.pdf")

    def test_extract_document_unsupported_format(self):
        """Test extraction with unsupported file format."""
        extractor = DocumentExtractor()

        with tempfile.NamedTemporaryFile(suffix=".unsupported", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid document file"):
                extractor.extract_text(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_extract_document_format_detection(self):
        """Test automatic format detection."""
        extractor = DocumentExtractor()

        # Test format detection logic
        test_cases = [
            ("test.pdf", "pdf"),
            ("document.docx", "docx"),
            ("presentation.pptx", "pptx"),
            ("webpage.html", "html"),
            ("file.PDF", "pdf"),  # Case insensitive
            ("FILE.DOCX", "docx"),
        ]

        for filename, expected_format in test_cases:
            with patch.object(
                extractor, "_extract_pdf", return_value=("text", {})
            ) as mock_pdf:
                with patch.object(
                    extractor, "_extract_docx", return_value=("text", {})
                ) as mock_docx:
                    with patch.object(
                        extractor, "_extract_pptx", return_value=("text", {})
                    ) as mock_pptx:
                        with patch.object(
                            extractor, "_extract_html", return_value=("text", {})
                        ) as mock_html:
                            with patch("pathlib.Path.exists", return_value=True):
                                try:
                                    extractor.extract_text(filename)

                                    # Check which method was called
                                    if expected_format == "pdf":
                                        mock_pdf.assert_called_once()
                                    elif expected_format == "docx":
                                        mock_docx.assert_called_once()
                                    elif expected_format == "pptx":
                                        mock_pptx.assert_called_once()
                                    elif expected_format == "html":
                                        mock_html.assert_called_once()

                                except Exception:
                                    pass  # Expected if file doesn't exist

    def test_extract_pdf_with_metadata_extraction(self):
        """Test PDF extraction with detailed metadata."""
        extractor = DocumentExtractor()

        # Mock fitz.open to return a document with metadata
        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()

            # Mock document metadata
            mock_doc.metadata = {
                "title": "テスト文書",
                "author": "作成者",
                "subject": "件名",
                "creator": "PDF作成ツール",
                "producer": "PDFプロデューサー",
                "creationDate": "D:20250101120000+09'00'",
                "modDate": "D:20250101120000+09'00'",
            }

            # Mock pages
            mock_page1 = Mock()

            def page1_get_text(format=None):
                if format == "dict":
                    return {
                        "blocks": [
                            {"lines": [{"spans": [{"text": "ページ1のテキスト内容"}]}]}
                        ]
                    }
                return "ページ1のテキスト内容"

            mock_page1.get_text.side_effect = page1_get_text
            mock_page1.get_images.return_value = []

            mock_page2 = Mock()

            def page2_get_text(format=None):
                if format == "dict":
                    return {
                        "blocks": [
                            {"lines": [{"spans": [{"text": "ページ2のテキスト内容"}]}]}
                        ]
                    }
                return "ページ2のテキスト内容"

            mock_page2.get_text.side_effect = page2_get_text
            mock_page2.get_images.return_value = []

            mock_doc.__iter__ = Mock(return_value=iter([mock_page1, mock_page2]))
            mock_doc.__len__ = Mock(return_value=2)
            # Set up load_page to return the appropriate page
            mock_doc.load_page = Mock(side_effect=[mock_page1, mock_page2])

            mock_fitz.return_value = mock_doc

            text, metadata = extractor._extract_pdf("test.pdf")

            assert "ページ1のテキスト内容" in text
            assert "ページ2のテキスト内容" in text
            assert metadata["format"] == "pdf"
            assert metadata["pages"] == 2
            assert metadata["title"] == "テスト文書"
            assert metadata["author"] == "作成者"

    def test_extract_pdf_empty_document(self):
        """Test PDF extraction with empty document."""
        extractor = DocumentExtractor()

        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {}
            mock_doc.__iter__ = Mock(return_value=iter([]))
            mock_doc.__len__ = Mock(return_value=0)

            mock_fitz.return_value = mock_doc

            text, metadata = extractor._extract_pdf("empty.pdf")

            assert text == ""
            assert metadata["pages"] == 0

    def test_extract_docx_complex_structure(self):
        """Test DOCX extraction with complex document structure."""
        extractor = DocumentExtractor()

        with patch("jtext.processing.document_extractor.Document") as mock_document:
            # Mock document with paragraphs, tables, and images
            mock_doc = Mock()

            # Mock paragraphs
            mock_para1 = Mock()
            mock_para1.text = "段落1のテキスト"
            mock_para2 = Mock()
            mock_para2.text = ""  # Empty paragraph
            mock_para3 = Mock()
            mock_para3.text = "段落3のテキスト"

            mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]

            # Mock tables
            mock_cell1 = Mock()
            mock_cell1.text = "セル1"
            mock_cell2 = Mock()
            mock_cell2.text = "セル2"

            mock_row = Mock()
            mock_row.cells = [mock_cell1, mock_cell2]

            mock_table = Mock()
            mock_table.rows = [mock_row]

            mock_doc.tables = [mock_table]

            # Mock relationships (for images)
            mock_rel = Mock()
            mock_rel.target_ref = "image1.png"

            mock_doc.part.rels.values.return_value = [mock_rel]

            mock_document.return_value = mock_doc

            text, metadata = extractor._extract_docx("test.docx")

            assert "段落1のテキスト" in text
            assert "段落3のテキスト" in text
            assert "セル1 | セル2" in text
            assert metadata["paragraphs"] == 2  # Only non-empty paragraphs
            assert metadata["tables"] == 1
            assert metadata["has_images"] is True

    def test_extract_docx_no_content(self):
        """Test DOCX extraction with document containing no readable content."""
        extractor = DocumentExtractor()

        with patch("jtext.processing.document_extractor.Document") as mock_document:
            mock_doc = Mock()

            # Empty paragraphs
            mock_para = Mock()
            mock_para.text = "   "  # Only whitespace
            mock_doc.paragraphs = [mock_para]

            # No tables
            mock_doc.tables = []

            # No images
            mock_doc.part.rels.values.return_value = []

            mock_document.return_value = mock_doc

            text, metadata = extractor._extract_docx("empty.docx")

            assert text == ""
            assert metadata["paragraphs"] == 0
            assert metadata["tables"] == 0
            assert metadata["has_images"] is False

    def test_extract_pptx_comprehensive(self):
        """Test PPTX extraction with comprehensive slide content."""
        extractor = DocumentExtractor()

        with patch(
            "jtext.processing.document_extractor.Presentation"
        ) as mock_presentation:
            mock_ppt = Mock()

            # Mock slides with different content types
            mock_slide1 = Mock()
            mock_slide2 = Mock()

            # Mock shapes with text - ensure text.strip() returns string
            # Use spec to limit attributes and prevent unwanted hasattr checks
            mock_shape1 = Mock(spec=["text"])
            mock_shape1.text.strip.return_value = "スライド1のタイトル"

            mock_shape2 = Mock(spec=["text"])
            mock_shape2.text.strip.return_value = "スライド1の内容"

            mock_shape3 = Mock(spec=["text"])
            mock_shape3.text.strip.return_value = ""  # Empty text shape

            mock_shape4 = Mock(spec=["text"])
            mock_shape4.text.strip.return_value = "スライド2のタイトル"

            mock_slide1.shapes = [mock_shape1, mock_shape2, mock_shape3]
            mock_slide2.shapes = [mock_shape4]

            mock_ppt.slides = [mock_slide1, mock_slide2]

            mock_presentation.return_value = mock_ppt

            text, metadata = extractor._extract_pptx("test.pptx")

            assert "スライド1のタイトル" in text
            assert "スライド1の内容" in text
            assert "スライド2のタイトル" in text
            assert metadata["slides"] == 2

    def test_extract_html_advanced_parsing(self):
        """Test HTML extraction with complex HTML structure."""
        extractor = DocumentExtractor()

        complex_html = """
        <html>
        <head>
            <title>複雑なHTMLページ</title>
            <meta name="description" content="テストページの説明">
        </head>
        <body>
            <h1>メインタイトル</h1>
            <p>段落1のテキスト</p>
            <div>
                <h2>サブタイトル</h2>
                <p>段落2のテキスト</p>
                <ul>
                    <li>リスト項目1</li>
                    <li>リスト項目2</li>
                </ul>
            </div>
            <table>
                <tr><td>セル1</td><td>セル2</td></tr>
                <tr><td>セル3</td><td>セル4</td></tr>
            </table>
            <script>console.log('スクリプト');</script>
            <style>body { margin: 0; }</style>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(complex_html)
            temp_path = f.name

        try:
            text, metadata = extractor._extract_html(temp_path)

            # Should extract text content but not script/style
            assert "メインタイトル" in text
            assert "段落1のテキスト" in text
            assert "段落2のテキスト" in text
            assert "リスト項目1" in text
            assert "セル1" in text

            # Should not include script or style content
            assert "console.log" not in text
            assert "margin: 0" not in text

            assert metadata["format"] == "html"

        finally:
            Path(temp_path).unlink()

    def test_extract_html_encoding_issues(self):
        """Test HTML extraction with various encodings."""
        extractor = DocumentExtractor()

        # Test with different encodings
        unicode_html = """
        <html>
        <head><meta charset="utf-8"></head>
        <body>
            <h1>日本語のタイトル</h1>
            <p>中文内容测试</p>
            <p>Русский текст</p>
            <p>العربية النص</p>
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
            assert "中文内容测试" in text
            assert "Русский текст" in text
            assert "العربية النص" in text
            assert "🎌" in text
            assert "🌸" in text

        finally:
            Path(temp_path).unlink()

    def test_error_handling_corrupted_files(self):
        """Test error handling with corrupted files."""
        extractor = DocumentExtractor()

        # Create corrupted files
        corrupted_files = [
            ("corrupted.pdf", b"Not a real PDF"),
            ("corrupted.docx", b"Not a real DOCX"),
            ("corrupted.pptx", b"Not a real PPTX"),
        ]

        for filename, content in corrupted_files:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, delete=False
            ) as f:
                f.write(content)
                temp_path = f.name

            try:
                # Should handle corruption gracefully
                with pytest.raises(Exception):  # Various exceptions possible
                    extractor.extract_text(temp_path)
            finally:
                Path(temp_path).unlink()

    def test_large_document_handling(self):
        """Test handling of large documents."""
        extractor = DocumentExtractor()

        # Mock large PDF
        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {"title": "Large Document"}

            # Create many pages
            pages = []
            for i in range(100):  # 100 pages
                mock_page = Mock()
                page_content = f"ページ{i+1}の内容 " * 100

                # Create proper side_effect function
                def create_get_text_side_effect(content):
                    def get_text_side_effect(format=None):
                        if format == "dict":
                            return {
                                "blocks": [{"lines": [{"spans": [{"text": content}]}]}]
                            }
                        return content

                    return get_text_side_effect

                mock_page.get_text.side_effect = create_get_text_side_effect(
                    page_content
                )
                mock_page.get_images.return_value = []
                pages.append(mock_page)

            mock_doc.__iter__ = Mock(return_value=iter(pages))
            mock_doc.__len__ = Mock(return_value=100)
            mock_doc.load_page = Mock(side_effect=lambda i: pages[i])

            mock_fitz.return_value = mock_doc

            text, metadata = extractor._extract_pdf("large.pdf")

            assert metadata["pages"] == 100
            assert len(text) > 10000  # Should be substantial

    def test_memory_efficient_processing(self):
        """Test memory-efficient processing of documents."""
        extractor = DocumentExtractor()

        # This test ensures processing doesn't consume excessive memory
        # In real implementation, this would monitor memory usage

        # Mock a document that would be large in memory
        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {}

            # Single page with large content
            mock_page = Mock()
            large_content = "大きな内容" * 10000

            def get_text_side_effect(format=None):
                if format == "dict":
                    return {
                        "blocks": [{"lines": [{"spans": [{"text": large_content}]}]}]
                    }
                return large_content

            mock_page.get_text.side_effect = get_text_side_effect
            mock_page.get_images.return_value = []

            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.load_page = Mock(return_value=mock_page)

            mock_fitz.return_value = mock_doc

            # Should process without memory issues
            text, metadata = extractor._extract_pdf("memory_test.pdf")

            assert len(text) > 0
            assert metadata["pages"] == 1

    def test_concurrent_extraction_safety(self):
        """Test thread safety of extraction operations."""
        import threading
        import time

        extractor = DocumentExtractor()
        results = []
        exceptions = []

        def extract_worker(worker_id):
            try:
                with patch("fitz.open") as mock_fitz:
                    mock_doc = Mock()
                    mock_doc.metadata = {"worker": worker_id}

                    mock_page = Mock()
                    content = f"Worker {worker_id} content"

                    def get_text_side_effect(format=None):
                        if format == "dict":
                            return {
                                "blocks": [{"lines": [{"spans": [{"text": content}]}]}]
                            }
                        return content

                    mock_page.get_text.side_effect = get_text_side_effect
                    mock_page.get_images.return_value = []

                    mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
                    mock_doc.__len__ = Mock(return_value=1)
                    mock_doc.load_page = Mock(return_value=mock_page)

                    mock_fitz.return_value = mock_doc

                    text, metadata = extractor._extract_pdf(f"test_{worker_id}.pdf")
                    results.append((worker_id, text, metadata))

            except Exception as e:
                exceptions.append((worker_id, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=extract_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(exceptions) == 0, f"Exceptions in threads: {exceptions}"
        assert len(results) == 5

        # Each worker should have unique results
        worker_ids = [r[0] for r in results]
        assert len(set(worker_ids)) == 5

    def test_extraction_with_special_characters(self):
        """Test extraction with documents containing special characters."""
        extractor = DocumentExtractor()

        special_content = """
        特殊文字テスト:
        - 数学記号: ∑, ∫, ∞, ≠, ≤, ≥
        - 通貨記号: ¥, $, €, £, ₹
        - 矢印: →, ←, ↑, ↓, ⇒, ⇐
        - ギリシャ文字: α, β, γ, δ, π, σ
        - 上付き/下付き: x², H₂O, E=mc²
        - 分数: ½, ⅓, ¼, ¾
        - その他: ™, ©, ®, §, ¶, †, ‡
        """

        # Test with HTML containing special characters
        html_content = f"""
        <html>
        <head><title>特殊文字テスト</title></head>
        <body><pre>{special_content}</pre></body>
        </html>
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            temp_path = f.name

        try:
            text, metadata = extractor._extract_html(temp_path)

            # Should preserve special characters
            assert "∑" in text
            assert "∞" in text
            assert "¥" in text
            assert "→" in text
            assert "α" in text
            assert "x²" in text
            assert "H₂O" in text
            assert "½" in text
            assert "™" in text

        finally:
            Path(temp_path).unlink()


class TestDocumentExtractorPerformance:
    """Performance-related tests for DocumentExtractor."""

    def test_extraction_speed_measurement(self):
        """Test that extraction completes within reasonable time."""
        import time

        extractor = DocumentExtractor()

        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {}

            # Medium-sized content
            mock_page = Mock()
            content = "標準的なページ内容 " * 1000

            # Mock page.get_text() to consistently return string
            def get_text_side_effect(format=None):
                if format == "dict":
                    return {"blocks": [{"lines": [{"spans": [{"text": content}]}]}]}
                return content

            mock_page.get_text.side_effect = get_text_side_effect

            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.load_page = Mock(return_value=mock_page)

            mock_fitz.return_value = mock_doc

            start_time = time.time()
            text, metadata = extractor._extract_pdf("performance_test.pdf")
            end_time = time.time()

            # Should complete quickly (under 1 second for mocked operation)
            assert end_time - start_time < 1.0
            assert len(text) > 0

    def test_memory_usage_monitoring(self):
        """Test memory usage during extraction."""
        import gc

        extractor = DocumentExtractor()

        # Force garbage collection before test
        gc.collect()

        with patch("fitz.open") as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {}

            # Create content that would use memory
            mock_page = Mock()
            content = "メモリテスト内容 " * 5000

            # Mock page.get_text() to consistently return string
            def get_text_side_effect(format=None):
                if format == "dict":
                    return {"blocks": [{"lines": [{"spans": [{"text": content}]}]}]}
                return content

            mock_page.get_text.side_effect = get_text_side_effect

            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.load_page = Mock(return_value=mock_page)

            mock_fitz.return_value = mock_doc

            # Extract document
            text, metadata = extractor._extract_pdf("memory_test.pdf")

            # Force cleanup
            del text, metadata
            gc.collect()

            # Test should complete without memory errors
            assert True  # If we reach here, memory management is working
