"""
Input validation utilities for jtext system.

This module provides validation functions for input files,
ensuring they meet the requirements for processing.
"""

from pathlib import Path
from typing import List, Set

from .logging import get_logger


logger = get_logger(__name__)


# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
}
SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".docx", ".pptx", ".html", ".htm"}
SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mov"}

# Maximum file size in MB
MAX_FILE_SIZE_MB: float = 2048.0  # 2GB as per requirements


def validate_image_file(file_path: str) -> bool:
    """
    Validate that a file is a supported image format and within size limits.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if file is valid, False otherwise
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False

    # Check file extension
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.error(f"Unsupported image format: {path.suffix}")
        return False

    # Check file size
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            return False
    except OSError as e:
        logger.error(f"Cannot access file {file_path}: {e}")
        return False

    logger.debug(f"Image file validation passed: {file_path}")
    return True


def validate_document_file(file_path: str) -> bool:
    """
    Validate that a file is a supported document format and within size limits.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if file is valid, False otherwise
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False

    # Check file extension
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        logger.error(f"Unsupported document format: {path.suffix}")
        return False

    # Check file size
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            return False
    except OSError as e:
        logger.error(f"Cannot access file {file_path}: {e}")
        return False

    logger.debug(f"Document file validation passed: {file_path}")
    return True


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that a file is a supported audio/video format and within size limits.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if file is valid, False otherwise
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False

    # Check file extension
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.error(f"Unsupported audio/video format: {path.suffix}")
        return False

    # Check file size
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            return False
    except OSError as e:
        logger.error(f"Cannot access file {file_path}: {e}")
        return False

    logger.debug(f"Audio file validation passed: {file_path}")
    return True


def validate_file_list(file_paths: List[str], file_type: str = "image") -> List[str]:
    """
    Validate a list of files and return valid ones.

    Args:
        file_paths: List of file paths to validate
        file_type: Type of files to validate ('image', 'document', 'audio')

    Returns:
        List of valid file paths
    """
    valid_files = []

    for file_path in file_paths:
        if file_type == "image" and validate_image_file(file_path):
            valid_files.append(file_path)
        elif file_type == "document" and validate_document_file(file_path):
            valid_files.append(file_path)
        elif file_type == "audio" and validate_audio_file(file_path):
            valid_files.append(file_path)
        else:
            logger.warning(f"Skipping invalid {file_type} file: {file_path}")

    logger.info(f"Validated {len(valid_files)}/{len(file_paths)} {file_type} files")
    return valid_files
