"""
Pytest configuration and fixtures for jtext tests.

This module provides common test fixtures and configuration
for the jtext test suite.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image file."""
    # Create a simple test image
    img_array = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
    img = Image.fromarray(img_array)

    image_path = temp_dir / "test_image.png"
    img.save(image_path)

    return str(image_path)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    text_path = temp_dir / "test.txt"
    text_path.write_text("これはテストテキストです。", encoding="utf-8")

    return str(text_path)


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a sample PDF file (empty for testing)."""
    pdf_path = temp_dir / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake pdf content\n%%EOF")

    return str(pdf_path)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file (empty for testing)."""
    audio_path = temp_dir / "test.mp3"
    audio_path.write_bytes(b"fake audio data")

    return str(audio_path)


@pytest.fixture
def mock_ocr_result():
    """Create a mock OCR processing result."""
    from jtext.core.ocr_hybrid import ProcessingResult

    return ProcessingResult(
        source_path="/test/image.png",
        text="テストテキスト",
        confidence=0.85,
        processing_time=5.2,
        memory_usage=128.5,
        corrections_applied=3,
        correction_ratio=0.02,
    )


@pytest.fixture
def mock_image():
    """Create a mock PIL Image object."""
    img_array = np.ones((100, 200, 3), dtype=np.uint8) * 255
    return Image.fromarray(img_array)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in test_* modules
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add slow marker to tests that might take time
        if any(
            keyword in item.nodeid.lower() for keyword in ["ocr", "process", "file"]
        ):
            item.add_marker(pytest.mark.slow)
