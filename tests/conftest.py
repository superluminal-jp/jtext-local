"""
Test configuration and fixtures.
"""

import tempfile
import pytest
from pathlib import Path
from typing import Generator

from jtext.domain import FilePath


@pytest.fixture
def temp_image_file() -> Generator[str, None, None]:
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"fake image content")
        f.flush()
        yield f.name
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_audio_file() -> Generator[str, None, None]:
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio content")
        f.flush()
        yield f.name
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def test_file_path(temp_image_file: str) -> FilePath:
    """Create a test FilePath using a temporary file."""
    return FilePath(temp_image_file)


@pytest.fixture
def test_audio_file_path(temp_audio_file: str) -> FilePath:
    """Create a test audio FilePath using a temporary file."""
    return FilePath(temp_audio_file)
