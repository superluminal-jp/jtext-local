"""
Core utilities and shared components.

This module contains shared utilities, constants, and common functionality
that can be used across all layers of the application.
"""

import os
import sys
import json
import uuid
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

# ============================================================================
# Constants
# ============================================================================

# Application metadata
APP_NAME = "jtext"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Japanese Text Processing System"

# Supported file types
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx"}

# Default configurations
DEFAULT_OCR_LANGUAGE = "jpn+eng"
DEFAULT_ASR_MODEL = "base"
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_VISION_MODEL = "llava:7b"

# Processing limits
MAX_FILE_SIZE_MB = 100
MAX_PROCESSING_TIME_SECONDS = 300
MAX_RETRY_ATTEMPTS = 3

# ============================================================================
# Utility Functions
# ============================================================================


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes."""
    return Path(file_path).stat().st_size / (1024 * 1024)


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Check if file type is supported."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    return (
        suffix in SUPPORTED_IMAGE_EXTENSIONS
        or suffix in SUPPORTED_AUDIO_EXTENSIONS
        or suffix in SUPPORTED_DOCUMENT_EXTENSIONS
    )


def get_mime_type(file_path: Union[str, Path]) -> Optional[str]:
    """Get MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely serialize data to JSON with proper error handling."""
    try:
        return json.dumps(data, default=str, **kwargs)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"JSON serialization failed: {str(e)}"}, **kwargs)


def safe_json_loads(data: str, **kwargs) -> Any:
    """Safely deserialize JSON data with proper error handling."""
    try:
        return json.loads(data, **kwargs)
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"JSON deserialization failed: {str(e)}")


# ============================================================================
# Generic Result Type
# ============================================================================

T = TypeVar("T")
E = TypeVar("E")


class Result(Generic[T, E]):
    """Generic Result type for error handling."""

    def __init__(self, value: T = None, error: E = None):
        # Determine if this is Ok or Err based on which parameter is explicitly provided
        if error is not None:
            # This is an Err case
            if value is not None:
                raise ValueError("Result cannot have both value and error")
            self._value = None
            self._error = error
            self._is_ok = False
        else:
            # This is an Ok case (including Ok(None))
            self._value = value
            self._error = None
            self._is_ok = True

    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._is_ok

    @property
    def is_err(self) -> bool:
        """Check if result is an error."""
        return not self._is_ok

    def unwrap(self) -> T:
        """Unwrap the value, raise if error."""
        if self._is_ok:
            return self._value
        raise RuntimeError(f"Attempted to unwrap error: {self._error}")

    def unwrap_err(self) -> E:
        """Unwrap the error, raise if success."""
        if not self._is_ok:
            return self._error
        raise RuntimeError(f"Attempted to unwrap error from success: {self._value}")

    def map(self, func: Callable[[T], Any]) -> "Result":
        """Map function over successful value."""
        if self._is_ok:
            try:
                return Ok(func(self._value))
            except Exception as e:
                return Err(e)
        return self

    def map_err(self, func: Callable[[E], Any]) -> "Result":
        """Map function over error."""
        if not self._is_ok:
            try:
                return Err(func(self._error))
            except Exception as e:
                return Err(e)
        return self

    def and_then(self, func: Callable[[T], "Result"]) -> "Result":
        """Chain operations on successful values."""
        if self._is_ok:
            return func(self._value)
        return self

    def or_else(self, func: Callable[[E], "Result"]) -> "Result":
        """Handle errors."""
        if not self._is_ok:
            return func(self._error)
        return self


def Ok(value: T = None) -> Result[T, E]:
    """Create a successful result."""
    return Result(value=value)


def Err(error: E) -> Result[T, E]:
    """Create an error result."""
    return Result(error=error)


# ============================================================================
# Logging Configuration
# ============================================================================


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


def setup_logging(config: LogConfig = None) -> logging.Logger:
    """Setup application logging."""
    if config is None:
        config = LogConfig()

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(getattr(logging, config.level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper()))
    console_formatter = logging.Formatter(config.format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if config.file_path:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_formatter = logging.Formatter(config.format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# Performance Monitoring
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    def finish(self) -> None:
        """Mark the end of performance measurement."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class PerformanceMonitor:
    """Performance monitoring context manager."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.metrics = PerformanceMetrics()

    def __enter__(self) -> "PerformanceMonitor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.metrics.finish()
        logger = logging.getLogger(APP_NAME)
        logger.info(f"Performance [{self.name}]: {self.metrics.duration_seconds:.2f}s")


# ============================================================================
# Configuration Management
# ============================================================================


@dataclass
class AppConfig:
    """Application configuration."""

    # Processing settings
    max_file_size_mb: float = MAX_FILE_SIZE_MB
    max_processing_time_seconds: int = MAX_PROCESSING_TIME_SECONDS
    max_retry_attempts: int = MAX_RETRY_ATTEMPTS

    # Model settings
    default_ocr_language: str = DEFAULT_OCR_LANGUAGE
    default_asr_model: str = DEFAULT_ASR_MODEL
    default_llm_model: str = DEFAULT_LLM_MODEL
    default_vision_model: str = DEFAULT_VISION_MODEL

    # Output settings
    output_directory: str = "./out"
    create_timestamped_dirs: bool = True

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_file_path(file_path: Union[str, Path]) -> Result[Path, str]:
    """Validate file path."""
    try:
        path = Path(file_path)
        if not path.exists():
            return Err(f"File does not exist: {file_path}")
        if not path.is_file():
            return Err(f"Path is not a file: {file_path}")
        if not is_supported_file(path):
            return Err(f"Unsupported file type: {file_path}")
        if get_file_size_mb(path) > MAX_FILE_SIZE_MB:
            return Err(
                f"File too large: {get_file_size_mb(path):.1f}MB > {MAX_FILE_SIZE_MB}MB"
            )
        return Ok(path)
    except Exception as e:
        return Err(f"File validation error: {str(e)}")


def validate_output_directory(output_dir: Union[str, Path]) -> Result[Path, str]:
    """Validate and create output directory."""
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            return Err(f"Output path is not a directory: {output_dir}")
        return Ok(path)
    except Exception as e:
        return Err(f"Output directory validation error: {str(e)}")


# ============================================================================
# Export all public components
# ============================================================================

__all__ = [
    # Constants
    "APP_NAME",
    "APP_VERSION",
    "APP_DESCRIPTION",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "DEFAULT_OCR_LANGUAGE",
    "DEFAULT_ASR_MODEL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_VISION_MODEL",
    "MAX_FILE_SIZE_MB",
    "MAX_PROCESSING_TIME_SECONDS",
    "MAX_RETRY_ATTEMPTS",
    # Utility functions
    "generate_id",
    "get_file_hash",
    "get_file_size_mb",
    "is_supported_file",
    "get_mime_type",
    "ensure_directory",
    "safe_json_dumps",
    "safe_json_loads",
    # Result type
    "Result",
    "Ok",
    "Err",
    # Logging
    "LogConfig",
    "setup_logging",
    # Performance
    "PerformanceMetrics",
    "PerformanceMonitor",
    # Configuration
    "AppConfig",
    # Validation
    "validate_file_path",
    "validate_output_directory",
]
