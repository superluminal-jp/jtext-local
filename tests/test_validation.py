"""
Tests for input validation utilities.

This module contains unit tests for file validation functions
and supported format checking.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from jtext.utils.validation import (
    validate_image_file,
    validate_document_file,
    validate_audio_file,
    validate_file_list,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_AUDIO_EXTENSIONS,
    MAX_FILE_SIZE_MB,
)


class TestFileValidation:
    """Test cases for file validation functions."""

    def test_validate_image_file_success(self):
        """Test successful image file validation."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a small test file
            Path(tmp_path).write_bytes(b"fake image data")

            result = validate_image_file(tmp_path)
            assert result is True

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_image_file_nonexistent(self):
        """Test validation of nonexistent file."""
        result = validate_image_file("/nonexistent/file.png")
        assert result is False

    def test_validate_image_file_unsupported_format(self):
        """Test validation of unsupported image format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            Path(tmp_path).write_bytes(b"fake data")

            result = validate_image_file(tmp_path)
            assert result is False

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("jtext.utils.validation.Path.stat")
    def test_validate_image_file_too_large(self, mock_stat):
        """Test validation of file that's too large."""
        mock_stat.return_value.st_size = (MAX_FILE_SIZE_MB + 1) * 1024 * 1024

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = validate_image_file(tmp_path)
            assert result is False

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_document_file_success(self):
        """Test successful document file validation."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            Path(tmp_path).write_bytes(b"fake pdf data")

            result = validate_document_file(tmp_path)
            assert result is True

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_audio_file_success(self):
        """Test successful audio file validation."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            Path(tmp_path).write_bytes(b"fake audio data")

            result = validate_audio_file(tmp_path)
            assert result is True

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_file_list(self):
        """Test file list validation."""
        # Create test files
        test_files = []
        for ext in [".png", ".jpg", ".pdf", ".mp3", ".xyz"]:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
                Path(tmp_path).write_bytes(b"fake data")
                test_files.append(tmp_path)

        try:
            # Test image validation
            valid_images = validate_file_list(test_files, "image")
            assert len(valid_images) == 2  # .png and .jpg

            # Test document validation
            valid_docs = validate_file_list(test_files, "document")
            assert len(valid_docs) == 1  # .pdf

            # Test audio validation
            valid_audio = validate_file_list(test_files, "audio")
            assert len(valid_audio) == 1  # .mp3

        finally:
            for tmp_path in test_files:
                Path(tmp_path).unlink(missing_ok=True)


class TestSupportedFormats:
    """Test cases for supported format constants."""

    def test_supported_image_extensions(self):
        """Test supported image extensions."""
        expected = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"}
        assert SUPPORTED_IMAGE_EXTENSIONS == expected

    def test_supported_document_extensions(self):
        """Test supported document extensions."""
        expected = {".pdf", ".docx", ".pptx", ".html", ".htm"}
        assert SUPPORTED_DOCUMENT_EXTENSIONS == expected

    def test_supported_audio_extensions(self):
        """Test supported audio extensions."""
        expected = {".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mov"}
        assert SUPPORTED_AUDIO_EXTENSIONS == expected

    def test_max_file_size(self):
        """Test maximum file size constant."""
        assert MAX_FILE_SIZE_MB == 2048.0
