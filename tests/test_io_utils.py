"""
Tests for I/O utilities module.

This module tests the file I/O utility functions including
directory management, result saving, and metadata operations.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from jtext.utils.io_utils import (
    ensure_output_dir,
    save_results,
    load_metadata,
    get_file_size_mb,
)


class TestEnsureOutputDir:
    """Test the ensure_output_dir function."""

    def test_ensure_output_dir_success(self):
        """Test successful directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_output"

            # Should create directory without error
            ensure_output_dir(output_dir)

            # Directory should exist
            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_ensure_output_dir_existing(self):
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)  # Use existing temp dir

            # Should work with existing directory
            ensure_output_dir(output_dir)

            assert output_dir.exists()

    def test_ensure_output_dir_nested(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "level1" / "level2" / "level3"

            ensure_output_dir(output_dir)

            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_ensure_output_dir_permission_error(self):
        """Test handling of permission errors."""
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            output_dir = Path("/non/existent/path")

            with pytest.raises(
                OSError, match="Cannot create or write to output directory"
            ):
                ensure_output_dir(output_dir)

    def test_ensure_output_dir_write_test(self):
        """Test write permission verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_write"

            # Mock touch to fail (simulate write permission issue)
            with patch(
                "pathlib.Path.touch", side_effect=PermissionError("No write access")
            ):
                with pytest.raises(OSError):
                    ensure_output_dir(output_dir)

    def test_ensure_output_dir_cleanup(self):
        """Test that write test file is properly cleaned up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_cleanup"

            ensure_output_dir(output_dir)

            # Test file should not exist after function completes
            test_file = output_dir / ".write_test"
            assert not test_file.exists()


class TestSaveResults:
    """Test the save_results function."""

    def test_save_results_success(self):
        """Test successful result saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_result"

            # Create mock result object
            mock_result = Mock()
            mock_result.text = "テストテキスト内容"
            mock_result.to_dict.return_value = {
                "text": "テストテキスト内容",
                "confidence": 0.95,
                "metadata": {"source": "test"},
            }

            save_results(mock_result, output_path)

            # Check text file was created
            text_file = output_path.with_suffix(".txt")
            assert text_file.exists()

            with open(text_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert content == "テストテキスト内容"

            # Check JSON file was created
            json_file = output_path.with_suffix(".json")
            assert json_file.exists()

            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                assert metadata["confidence"] == 0.95
                assert metadata["metadata"]["source"] == "test"

    def test_save_results_unicode_handling(self):
        """Test proper Unicode handling in saved files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "unicode_test"

            # Test with various Unicode characters
            unicode_text = "日本語テスト 中文测试 🎌 символы العربية"

            mock_result = Mock()
            mock_result.text = unicode_text
            mock_result.to_dict.return_value = {
                "text": unicode_text,
                "unicode_test": "特殊文字: ♥ ♦ ♣ ♠",
            }

            save_results(mock_result, output_path)

            # Verify Unicode characters are preserved
            text_file = output_path.with_suffix(".txt")
            with open(text_file, "r", encoding="utf-8") as f:
                saved_text = f.read()
                assert saved_text == unicode_text

            json_file = output_path.with_suffix(".json")
            with open(json_file, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)
                assert "特殊文字" in saved_metadata["unicode_test"]

    def test_save_results_file_error(self):
        """Test handling of file write errors."""
        # Use invalid path that should cause write error
        invalid_path = Path("/invalid/nonexistent/path/result")

        mock_result = Mock()
        mock_result.text = "test"
        mock_result.to_dict.return_value = {"test": "data"}

        with pytest.raises(Exception):
            save_results(mock_result, invalid_path)

    def test_save_results_json_serialization_error(self):
        """Test handling of JSON serialization errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "json_error_test"

            # Create result with non-serializable data
            mock_result = Mock()
            mock_result.text = "test text"
            mock_result.to_dict.side_effect = TypeError("Object not JSON serializable")

            with pytest.raises(TypeError):
                save_results(mock_result, output_path)

    def test_save_results_large_content(self):
        """Test saving large content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "large_content"

            # Create large text content
            large_text = "長いテキスト内容 " * 10000

            mock_result = Mock()
            mock_result.text = large_text
            mock_result.to_dict.return_value = {
                "text": large_text,
                "size": len(large_text),
            }

            save_results(mock_result, output_path)

            # Verify large content was saved correctly
            text_file = output_path.with_suffix(".txt")
            with open(text_file, "r", encoding="utf-8") as f:
                saved_text = f.read()
                assert len(saved_text) == len(large_text)

    def test_save_results_empty_content(self):
        """Test saving empty or minimal content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_content"

            mock_result = Mock()
            mock_result.text = ""
            mock_result.to_dict.return_value = {"text": "", "empty": True}

            save_results(mock_result, output_path)

            # Files should still be created
            text_file = output_path.with_suffix(".txt")
            json_file = output_path.with_suffix(".json")

            assert text_file.exists()
            assert json_file.exists()

            # Content should be empty
            with open(text_file, "r", encoding="utf-8") as f:
                assert f.read() == ""


class TestLoadMetadata:
    """Test the load_metadata function."""

    def test_load_metadata_success(self):
        """Test successful metadata loading."""
        test_metadata = {
            "text": "テストテキスト",
            "confidence": 0.89,
            "processing_time": 2.5,
            "metadata": {"source": "test.png", "language": "Japanese"},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(test_metadata, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            loaded_metadata = load_metadata(temp_path)

            assert loaded_metadata == test_metadata
            assert loaded_metadata["confidence"] == 0.89
            assert loaded_metadata["metadata"]["language"] == "Japanese"
        finally:
            temp_path.unlink()

    def test_load_metadata_file_not_found(self):
        """Test handling of missing metadata file."""
        nonexistent_file = Path("/nonexistent/metadata.json")

        with pytest.raises(FileNotFoundError):
            load_metadata(nonexistent_file)

    def test_load_metadata_invalid_json(self):
        """Test handling of invalid JSON content."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{ invalid json content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                load_metadata(temp_path)
        finally:
            temp_path.unlink()

    def test_load_metadata_empty_file(self):
        """Test handling of empty JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                load_metadata(temp_path)
        finally:
            temp_path.unlink()

    def test_load_metadata_unicode_content(self):
        """Test loading metadata with Unicode content."""
        unicode_metadata = {
            "japanese": "日本語テキスト",
            "chinese": "中文内容",
            "emoji": "🌸🎌📄",
            "special_chars": "→←↑↓",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(unicode_metadata, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            loaded_metadata = load_metadata(temp_path)

            assert loaded_metadata["japanese"] == "日本語テキスト"
            assert loaded_metadata["emoji"] == "🌸🎌📄"
        finally:
            temp_path.unlink()

    def test_load_metadata_complex_structure(self):
        """Test loading complex nested metadata structure."""
        complex_metadata = {
            "results": {
                "ocr": {
                    "text": "抽出されたテキスト",
                    "confidence": 0.92,
                    "corrections": ["訂正1", "訂正2"],
                },
                "vision": {
                    "document_type": "technical",
                    "layout": ["header", "body", "footer"],
                    "tables": True,
                },
            },
            "processing": {
                "steps": ["preprocess", "ocr", "correct"],
                "timing": [0.5, 2.1, 0.8],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(complex_metadata, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            loaded_metadata = load_metadata(temp_path)

            assert loaded_metadata["results"]["ocr"]["confidence"] == 0.92
            assert loaded_metadata["results"]["vision"]["tables"] is True
            assert len(loaded_metadata["processing"]["steps"]) == 3
        finally:
            temp_path.unlink()


class TestGetFileSizeMb:
    """Test the get_file_size_mb function."""

    def test_get_file_size_mb_success(self):
        """Test successful file size calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write approximately 1KB of data
            test_content = "A" * 1024
            f.write(test_content.encode("utf-8"))
            temp_path = Path(f.name)

        try:
            size_mb = get_file_size_mb(temp_path)

            # Should be approximately 0.001 MB (1KB)
            assert 0.0009 <= size_mb <= 0.002  # Allow for small variations
        finally:
            temp_path.unlink()

    def test_get_file_size_mb_large_file(self):
        """Test with larger file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write approximately 1MB of data
            chunk = "B" * 1024  # 1KB chunk
            for _ in range(1024):  # Write 1024 chunks = ~1MB
                f.write(chunk.encode("utf-8"))
            temp_path = Path(f.name)

        try:
            size_mb = get_file_size_mb(temp_path)

            # Should be approximately 1 MB
            assert 0.9 <= size_mb <= 1.1
        finally:
            temp_path.unlink()

    def test_get_file_size_mb_empty_file(self):
        """Test with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            size_mb = get_file_size_mb(temp_path)
            assert size_mb == 0.0
        finally:
            temp_path.unlink()

    def test_get_file_size_mb_nonexistent_file(self):
        """Test with non-existent file."""
        nonexistent_file = Path("/nonexistent/file.txt")

        size_mb = get_file_size_mb(nonexistent_file)
        assert size_mb == 0.0

    def test_get_file_size_mb_permission_error(self):
        """Test handling of permission errors."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Mock stat to raise permission error
            with patch(
                "pathlib.Path.stat", side_effect=PermissionError("Access denied")
            ):
                size_mb = get_file_size_mb(temp_path)
                assert size_mb == 0.0
        finally:
            temp_path.unlink()

    def test_get_file_size_mb_unicode_filename(self):
        """Test with Unicode filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            unicode_filename = "テストファイル_🎌.txt"
            file_path = Path(temp_dir) / unicode_filename

            # Create file with Unicode name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("日本語コンテンツ" * 100)

            size_mb = get_file_size_mb(file_path)
            assert size_mb > 0.0


class TestIOUtilsIntegration:
    """Integration tests for I/O utilities."""

    def test_complete_save_load_cycle(self):
        """Test complete save and load cycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "integration_test"

            # Create mock result
            mock_result = Mock()
            mock_result.text = "統合テストの内容です。"
            mock_result.to_dict.return_value = {
                "text": "統合テストの内容です。",
                "confidence": 0.95,
                "processing_time": 1.23,
                "source": "integration_test.png",
            }

            # Save results
            save_results(mock_result, output_path)

            # Load metadata back
            metadata_file = output_path.with_suffix(".json")
            loaded_metadata = load_metadata(metadata_file)

            # Verify round-trip integrity
            assert loaded_metadata["text"] == "統合テストの内容です。"
            assert loaded_metadata["confidence"] == 0.95
            assert loaded_metadata["processing_time"] == 1.23

    def test_directory_and_file_operations(self):
        """Test directory creation and file operations together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested output directory
            output_dir = Path(temp_dir) / "nested" / "output"
            ensure_output_dir(output_dir)

            # Save results to the directory
            output_path = output_dir / "test_file"

            mock_result = Mock()
            mock_result.text = "ディレクトリ統合テスト"
            mock_result.to_dict.return_value = {"text": "ディレクトリ統合テスト"}

            save_results(mock_result, output_path)

            # Verify files exist
            text_file = output_path.with_suffix(".txt")
            json_file = output_path.with_suffix(".json")

            assert text_file.exists()
            assert json_file.exists()

            # Check file sizes
            text_size = get_file_size_mb(text_file)
            json_size = get_file_size_mb(json_file)

            assert text_size > 0.0
            assert json_size > 0.0

    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "error_test"

            # Ensure directory exists
            ensure_output_dir(output_dir)

            # Test with various problematic scenarios
            problematic_results = [
                # Empty text
                {"text": "", "metadata": {}},
                # Very long text
                {"text": "長" * 10000, "metadata": {"size": "large"}},
                # Special characters
                {"text": "🎌📄🔤", "metadata": {"type": "emoji"}},
            ]

            for i, result_data in enumerate(problematic_results):
                mock_result = Mock()
                mock_result.text = result_data["text"]
                mock_result.to_dict.return_value = result_data

                output_path = output_dir / f"test_{i}"

                try:
                    save_results(mock_result, output_path)

                    # Verify files were created successfully
                    assert output_path.with_suffix(".txt").exists()
                    assert output_path.with_suffix(".json").exists()

                except Exception as e:
                    pytest.fail(f"Error recovery failed for test case {i}: {e}")
