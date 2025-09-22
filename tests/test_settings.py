"""
Tests for configuration settings module.

This module tests the Settings class which manages configuration
for the jtext application including default values, environment
variable overrides, and file-based configuration.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from jtext.config.settings import Settings


class TestSettings:
    """Test the Settings configuration class."""

    def test_settings_initialization(self):
        """Test basic settings initialization."""
        settings = Settings()
        assert settings is not None

        # Test default values
        assert settings.get("ocr", "default_language") == "jpn+eng"
        assert settings.get("llm", "default_model") == "gpt-oss"
        assert settings.get("processing", "max_file_size_mb") == 2048.0

    def test_settings_get_method(self):
        """Test the get method with different scenarios."""
        settings = Settings()

        # Test existing values
        assert settings.get("ocr", "default_language") == "jpn+eng"
        assert settings.get("llm", "temperature") == 0.1
        assert settings.get("processing", "max_concurrent_files") == 4

        # Test default values
        assert settings.get("nonexistent", "key", "default") == "default"
        assert settings.get("ocr", "nonexistent", 42) == 42

        # Test None default
        assert settings.get("nonexistent", "key") is None

    def test_settings_set_method(self):
        """Test the set method."""
        settings = Settings()

        # Test setting new values
        settings.set("ocr", "confidence_threshold", 0.8)
        assert settings.get("ocr", "confidence_threshold") == 0.8

        # Test overriding existing values
        settings.set("llm", "default_model", "llama2")
        assert settings.get("llm", "default_model") == "llama2"

        # Test setting in new section
        settings.set("new_section", "new_key", "new_value")
        assert settings.get("new_section", "new_key") == "new_value"

    def test_settings_to_dict(self):
        """Test accessing settings like dictionary."""
        settings = Settings()

        # Test that settings can be accessed
        assert settings.get("ocr", "default_language") == "jpn+eng"
        assert settings.get("llm", "default_model") == "gpt-oss"
        assert settings.get("processing", "max_file_size_mb") == 2048.0
        assert settings.get("logging", "level") == "INFO"

    @patch.dict(
        os.environ,
        {
            "JTEXT_OCR_LANG": "eng",
            "JTEXT_LLM_MODEL": "llama2",
            "JTEXT_MAX_FILE_SIZE": "1024",
            "JTEXT_MEMORY_LIMIT": "8192",
            "JTEXT_LOG_LEVEL": "DEBUG",
        },
    )
    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        settings = Settings()

        # Test string overrides
        assert settings.get("ocr", "default_language") == "eng"
        assert settings.get("llm", "default_model") == "llama2"
        assert settings.get("logging", "level") == "DEBUG"

        # Test numeric overrides
        assert settings.get("processing", "max_file_size_mb") == 1024.0
        assert settings.get("processing", "memory_limit_mb") == 8192.0

    @patch.dict(
        os.environ, {"JTEXT_OCR_CONFIDENCE": "0.85", "JTEXT_LLM_TEMPERATURE": "0.2"}
    )
    def test_float_environment_overrides(self):
        """Test float type environment variable overrides."""
        settings = Settings()

        # Note: These would need to be added to the env_mappings if they existed
        # For now, test the conversion logic with existing float fields
        pass  # This test would require modifying the env_mappings

    def test_settings_sections(self):
        """Test accessing different configuration sections."""
        settings = Settings()

        # Test OCR section
        ocr_lang = settings.get("ocr", "default_language")
        assert isinstance(ocr_lang, str)

        # Test LLM section
        llm_model = settings.get("llm", "default_model")
        assert isinstance(llm_model, str)

        # Test processing section
        max_size = settings.get("processing", "max_file_size_mb")
        assert isinstance(max_size, float)

        # Test logging section
        log_level = settings.get("logging", "level")
        assert isinstance(log_level, str)

    def test_settings_file_integration(self):
        """Test settings file loading (if config file exists)."""
        settings = Settings()

        # Test that settings can be loaded without errors
        # The actual file loading would be tested if config files existed
        assert settings.get("ocr", "default_language") is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_settings_without_environment(self):
        """Test settings behavior without environment variables."""
        settings = Settings()

        # Should use default values
        assert settings.get("ocr", "default_language") == "jpn+eng"
        assert settings.get("llm", "default_model") == "gpt-oss"
        assert settings.get("processing", "max_file_size_mb") == 2048.0

    def test_settings_boolean_conversion(self):
        """Test boolean value conversion from environment variables."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("anything_else", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"TEST_BOOL": env_value}):
                # This would test the boolean conversion logic
                # The actual implementation would need the env var in env_mappings
                pass

    def test_settings_immutability_after_init(self):
        """Test that settings maintain consistency after initialization."""
        settings = Settings()
        original_lang = settings.get("ocr", "default_language")

        # Modify settings
        settings.set("ocr", "default_language", "eng")
        assert settings.get("ocr", "default_language") == "eng"

        # Create new instance - should have original defaults
        new_settings = Settings()
        assert new_settings.get("ocr", "default_language") == original_lang

    def test_settings_numeric_validation(self):
        """Test numeric value handling."""
        settings = Settings()

        # Test integer values
        max_files = settings.get("processing", "max_concurrent_files")
        assert isinstance(max_files, int)
        assert max_files > 0

        # Test float values
        max_size = settings.get("processing", "max_file_size_mb")
        assert isinstance(max_size, float)
        assert max_size > 0.0

    def test_settings_error_handling(self):
        """Test error handling in settings operations."""
        settings = Settings()

        # Test accessing non-existent sections gracefully
        assert settings.get("nonexistent_section", "key") is None
        assert settings.get("nonexistent_section", "key", "default") == "default"

        # Test setting values doesn't crash
        try:
            settings.set("test_section", "test_key", "test_value")
            assert settings.get("test_section", "test_key") == "test_value"
        except Exception as e:
            pytest.fail(f"Setting value should not raise exception: {e}")


class TestSettingsIntegration:
    """Integration tests for Settings class."""

    def test_settings_with_all_sections(self):
        """Test that all expected configuration sections are present."""
        settings = Settings()

        # Test that each section has expected keys
        assert settings.get("ocr", "default_language") is not None
        assert settings.get("llm", "default_model") is not None
        assert settings.get("asr", "default_model") is not None
        assert settings.get("processing", "max_file_size_mb") is not None
        assert settings.get("logging", "level") is not None

    def test_settings_realistic_values(self):
        """Test that settings contain realistic default values."""
        settings = Settings()

        # OCR settings
        assert "jpn" in settings.get("ocr", "default_language")
        assert settings.get("ocr", "confidence_threshold") >= 0.0
        assert settings.get("ocr", "confidence_threshold") <= 1.0

        # LLM settings
        assert settings.get("llm", "temperature") >= 0.0
        assert settings.get("llm", "temperature") <= 2.0
        assert settings.get("llm", "max_tokens") > 0

        # Processing settings
        assert settings.get("processing", "max_file_size_mb") > 0
        assert settings.get("processing", "max_concurrent_files") > 0

    @patch.dict(
        os.environ,
        {
            "JTEXT_OCR_LANG": "jpn+eng+chi_sim",
            "JTEXT_LLM_MODEL": "gemma3:latest",
            "JTEXT_MAX_FILE_SIZE": "4096",
        },
    )
    def test_settings_production_scenario(self):
        """Test settings in a production-like scenario."""
        settings = Settings()

        # Verify environment overrides work
        assert settings.get("ocr", "default_language") == "jpn+eng+chi_sim"
        assert settings.get("llm", "default_model") == "gemma3:latest"
        assert settings.get("processing", "max_file_size_mb") == 4096.0

        # Verify non-overridden values remain default
        assert settings.get("llm", "temperature") == 0.1  # default value
