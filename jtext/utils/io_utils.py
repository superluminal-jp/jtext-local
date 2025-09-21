"""
File I/O utilities for jtext system.

This module provides utilities for file operations, output directory
management, and result serialization.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .logging import get_logger


logger = get_logger(__name__)


def ensure_output_dir(output_dir: Path) -> None:
    """
    Ensure the output directory exists and is writable.

    Args:
        output_dir: Path to the output directory

    Raises:
        OSError: If directory cannot be created or is not writable
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test write permissions
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()

        logger.debug(f"Output directory ready: {output_dir}")

    except Exception as e:
        logger.error(f"Failed to setup output directory {output_dir}: {e}")
        raise OSError(f"Cannot create or write to output directory: {output_dir}")


def save_results(result: Any, output_path: Path) -> None:
    """
    Save processing results to text and JSON files.

    Args:
        result: ProcessingResult object containing text and metadata
        output_path: Base path for output files (without extension)
    """
    try:
        # Save text content
        text_file = output_path.with_suffix(".txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(result.text)

        logger.debug(f"Saved text to: {text_file}")

        # Save metadata
        metadata_file = output_path.with_suffix(".json")
        metadata = result.to_dict()

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved metadata to: {metadata_file}")

    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def load_metadata(metadata_file: Path) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.

    Args:
        metadata_file: Path to the metadata JSON file

    Returns:
        Dictionary containing metadata

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.debug(f"Loaded metadata from: {metadata_file}")
        return metadata

    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file {metadata_file}: {e}")
        raise


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in megabytes
    """
    try:
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except OSError:
        return 0.0
