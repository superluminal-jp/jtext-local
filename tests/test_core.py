"""
Tests for core utilities and shared components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from jtext.core import (
    Result,
    Ok,
    Err,
    generate_id,
    get_file_hash,
    get_file_size_mb,
    is_supported_file,
    validate_file_path,
    validate_output_directory,
    setup_logging,
    PerformanceMonitor,
    AppConfig,
)


class TestResult:
    """Test Result type."""

    def test_ok_result(self):
        """Test successful result."""
        result = Ok("success")
        assert result.is_ok
        assert not result.is_err
        assert result.unwrap() == "success"

    def test_err_result(self):
        """Test error result."""
        result = Err("error")
        assert not result.is_ok
        assert result.is_err
        assert result.unwrap_err() == "error"

    def test_result_map(self):
        """Test result mapping."""
        result = Ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok
        assert mapped.unwrap() == 10

    def test_result_map_error(self):
        """Test result mapping with error."""
        result = Err("error")
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err
        assert mapped.unwrap_err() == "error"

    def test_result_and_then(self):
        """Test result chaining."""
        result = Ok(5)
        chained = result.and_then(lambda x: Ok(x * 2))
        assert chained.is_ok
        assert chained.unwrap() == 10


class TestUtilities:
    """Test utility functions."""

    def test_generate_id(self):
        """Test ID generation."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0

    def test_get_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            hash_value = get_file_hash(f.name)
            assert len(hash_value) == 64  # SHA-256 hex length

            Path(f.name).unlink()

    def test_get_file_size_mb(self):
        """Test file size calculation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            size_mb = get_file_size_mb(f.name)
            assert size_mb > 0

            Path(f.name).unlink()

    def test_is_supported_file(self):
        """Test supported file detection."""
        # Test supported files
        assert is_supported_file("test.jpg")
        assert is_supported_file("test.png")
        assert is_supported_file("test.wav")
        assert is_supported_file("test.mp3")
        assert is_supported_file("test.pdf")

        # Test unsupported files
        assert not is_supported_file("test.txt")
        assert not is_supported_file("test.exe")

    def test_validate_file_path(self):
        """Test file path validation."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test content")
            f.flush()

            result = validate_file_path(f.name)
            assert result.is_ok

            Path(f.name).unlink()

    def test_validate_file_path_nonexistent(self):
        """Test file path validation with nonexistent file."""
        result = validate_file_path("nonexistent.jpg")
        assert result.is_err
        assert "does not exist" in result.unwrap_err()

    def test_validate_output_directory(self):
        """Test output directory validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_output_directory(temp_dir)
            assert result.is_ok
            assert result.unwrap().exists()


class TestPerformanceMonitor:
    """Test performance monitoring."""

    def test_performance_monitor(self):
        """Test performance monitoring."""
        with PerformanceMonitor("test_operation") as monitor:
            import time

            time.sleep(0.1)  # Simulate work

        assert monitor.metrics.end_time is not None
        assert monitor.metrics.duration_seconds is not None
        assert monitor.metrics.duration_seconds >= 0.1


class TestAppConfig:
    """Test application configuration."""

    def test_app_config_defaults(self):
        """Test default configuration."""
        config = AppConfig()
        assert config.max_file_size_mb > 0
        assert config.max_processing_time_seconds > 0
        assert config.max_retry_attempts > 0

    def test_app_config_from_dict(self):
        """Test configuration from dictionary."""
        data = {
            "max_file_size_mb": 50.0,
            "max_processing_time_seconds": 120,
            "max_retry_attempts": 5,
        }
        config = AppConfig.from_dict(data)
        assert config.max_file_size_mb == 50.0
        assert config.max_processing_time_seconds == 120
        assert config.max_retry_attempts == 5

    def test_app_config_to_dict(self):
        """Test configuration to dictionary."""
        config = AppConfig()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert "max_file_size_mb" in data
        assert "max_processing_time_seconds" in data
        assert "max_retry_attempts" in data
