"""
Configuration settings for jtext system.

This module provides centralized configuration management for
model selection, processing parameters, and system settings.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os

from ..utils.logging import get_logger


logger = get_logger(__name__)


class Settings:
    """
    Centralized configuration management for jtext system.

    This class manages all configuration settings including model
    selection, processing parameters, and system preferences.
    """

    def __init__(self):
        """Initialize settings with default values."""
        self._settings = self._load_default_settings()
        self._load_environment_overrides()

    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            # OCR Settings
            "ocr": {
                "default_language": "jpn+eng",
                "tesseract_config": "--psm 6",
                "confidence_threshold": 0.6,
                "enable_preprocessing": True,
            },
            # LLM Settings
            "llm": {
                "default_model": "gpt-oss",
                "correction_enabled": False,
                "max_tokens": 2048,
                "temperature": 0.1,
            },
            # ASR Settings
            "asr": {
                "default_model": "kotoba",
                "language": "ja",
                "enable_correction": False,
            },
            # Processing Settings
            "processing": {
                "max_file_size_mb": 2048.0,
                "max_concurrent_files": 4,
                "memory_limit_mb": 8192,
                "timeout_seconds": 300,
            },
            # Output Settings
            "output": {
                "default_format": "txt",
                "include_metadata": True,
                "metadata_format": "json",
                "encoding": "utf-8",
            },
            # Logging Settings
            "logging": {
                "level": "INFO",
                "format": "structured",
                "max_file_size_mb": 10,
                "retention_days": 7,
            },
        }

    def _load_environment_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        env_mappings = {
            "JTEXT_OCR_LANG": ("ocr", "default_language"),
            "JTEXT_LLM_MODEL": ("llm", "default_model"),
            "JTEXT_ASR_MODEL": ("asr", "default_model"),
            "JTEXT_MAX_FILE_SIZE": ("processing", "max_file_size_mb"),
            "JTEXT_MEMORY_LIMIT": ("processing", "memory_limit_mb"),
            "JTEXT_LOG_LEVEL": ("logging", "level"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if key in ["max_file_size_mb", "memory_limit_mb"]:
                    value = float(value)
                elif key in ["max_concurrent_files", "timeout_seconds", "max_tokens"]:
                    value = int(value)
                elif key in ["confidence_threshold", "temperature"]:
                    value = float(value)
                elif key in [
                    "enable_preprocessing",
                    "correction_enabled",
                    "enable_correction",
                    "include_metadata",
                ]:
                    value = value.lower() in ("true", "1", "yes", "on")

                self._settings[section][key] = value
                logger.debug(f"Override from {env_var}: {section}.{key} = {value}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section name
            key: Configuration key name
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._settings.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            section: Configuration section name
            key: Configuration key name
            value: Value to set
        """
        if section not in self._settings:
            self._settings[section] = {}

        self._settings[section][key] = value
        logger.debug(f"Set configuration: {section}.{key} = {value}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Configuration section name

        Returns:
            Dictionary containing section configuration
        """
        return self._settings.get(section, {}).copy()

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate OCR settings
            ocr_lang = self.get("ocr", "default_language")
            if not isinstance(ocr_lang, str) or not ocr_lang:
                logger.error("Invalid OCR language setting")
                return False

            # Validate processing limits
            max_file_size = self.get("processing", "max_file_size_mb")
            if not isinstance(max_file_size, (int, float)) or max_file_size <= 0:
                logger.error("Invalid max file size setting")
                return False

            memory_limit = self.get("processing", "memory_limit_mb")
            if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
                logger.error("Invalid memory limit setting")
                return False

            logger.debug("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global settings instance
settings = Settings()
