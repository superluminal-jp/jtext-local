"""
Tests for the CLI module.

This module contains unit tests for the command-line interface
and command execution.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner

from jtext.cli import cli


class TestCLI:
    """Test cases for CLI functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Japanese Text Processing CLI System" in result.output

    def test_cli_verbose_option(self):
        """Test CLI verbose option."""
        result = self.runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_cli_output_dir_option(self):
        """Test CLI output directory option."""
        result = self.runner.invoke(cli, ["--output-dir", "/tmp/test", "--help"])
        assert result.exit_code == 0


class TestOCRCommand:
    """Test cases for OCR command."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_ocr_command_help(self):
        """Test OCR command help."""
        result = self.runner.invoke(cli, ["ocr", "--help"])
        assert result.exit_code == 0
        assert "Extract text from images" in result.output

    def test_ocr_command_no_files(self):
        """Test OCR command with no files specified."""
        result = self.runner.invoke(cli, ["ocr"])
        assert result.exit_code != 0
        assert "No image files specified" in result.output

    @patch("jtext.cli.HybridOCR")
    @patch("jtext.cli.save_results")
    def test_ocr_command_success(self, mock_save, mock_ocr_class):
        """Test successful OCR command execution."""
        # Setup mocks
        mock_ocr = Mock()
        mock_result = Mock()
        mock_result.text = "テストテキスト"
        mock_ocr.process_image.return_value = mock_result
        mock_ocr_class.return_value = mock_ocr

        # Create test image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image data")

        try:
            result = self.runner.invoke(cli, ["ocr", tmp_path])
            assert result.exit_code == 0
            assert "Completed:" in result.output

            # Verify OCR was called
            mock_ocr_class.assert_called_once()
            mock_ocr.process_image.assert_called_once_with(tmp_path)
            mock_save.assert_called_once()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.cli.HybridOCR")
    @patch("jtext.cli.validate_image_file")
    def test_ocr_command_with_correction(self, mock_validate, mock_ocr_class):
        """Test OCR command with LLM correction enabled."""
        # Create test image file first
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image data")

        # Setup mocks
        mock_validate.return_value = True
        mock_ocr = Mock()
        mock_result = Mock()
        mock_result.text = "修正されたテキスト"
        mock_result.confidence = 0.85
        mock_result.processing_time = 1.0
        mock_result.memory_usage = 10.0
        mock_result.corrections_applied = 2
        mock_result.correction_ratio = 0.1
        mock_result.source_path = tmp_path
        mock_result.to_dict.return_value = {
            "source": tmp_path,
            "type": "image",
            "timestamp": "2025-09-22T00:00:00Z",
            "processing": {"pipeline": ["tesseract", "llm_correction"]},
            "correction_stats": {"characters_changed": 2},
            "quality_metrics": {"character_count": 7},
        }
        mock_ocr.process_image.return_value = mock_result
        mock_ocr_class.return_value = mock_ocr

        try:
            result = self.runner.invoke(cli, ["ocr", "--llm-correct", tmp_path])
            if result.exit_code != 0:
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0

            # Verify OCR was called with correction enabled
            mock_ocr_class.assert_called_once_with(
                llm_model="gpt-oss", enable_correction=True
            )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.cli.HybridOCR")
    def test_ocr_command_error(self, mock_ocr_class):
        """Test OCR command with processing error."""
        # Setup mocks to raise exception
        mock_ocr = Mock()
        mock_ocr.process_image.side_effect = Exception("Processing failed")
        mock_ocr_class.return_value = mock_ocr

        # Create test image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image data")

        try:
            result = self.runner.invoke(cli, ["ocr", tmp_path])
            assert result.exit_code != 0
            assert "Error processing" in result.output

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestOtherCommands:
    """Test cases for other CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_ingest_command_help(self):
        """Test ingest command help."""
        result = self.runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "Extract text from structured documents" in result.output

    def test_ingest_command_implementation(self):
        """Test ingest command is now implemented."""
        # Create a test HTML file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as tmp:
            tmp.write("<html><body><p>Test content</p></body></html>")
            tmp_path = tmp.name

        try:
            result = self.runner.invoke(cli, ["ingest", tmp_path])
            # Should succeed or fail with proper error (not "not implemented")
            assert "not yet implemented in MVP" not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_transcribe_command_help(self):
        """Test transcribe command help."""
        result = self.runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "Transcribe audio/video files" in result.output

    def test_transcribe_command_implementation(self):
        """Test transcribe command is now implemented."""
        # Create a test audio file (fake)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"fake audio data")
            tmp_path = tmp.name

        try:
            result = self.runner.invoke(cli, ["transcribe", tmp_path])
            # Should fail with proper error (not "not implemented")
            assert "not yet implemented in MVP" not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_chat_command_help(self):
        """Test chat command help."""
        result = self.runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Interact with LLM" in result.output

    def test_chat_command_not_implemented(self):
        """Test chat command shows not implemented message."""
        result = self.runner.invoke(cli, ["chat", "--prompt", "test"])
        assert result.exit_code == 0
        assert "not yet implemented in MVP" in result.output
