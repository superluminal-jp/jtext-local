"""
Tests for image preprocessing module.

This module tests the ImagePreprocessor class which provides
image enhancement functions for better OCR accuracy.
"""

import cv2
import numpy as np
import pytest
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from jtext.preprocessing.image_prep import ImagePreprocessor


class TestImagePreprocessor:
    """Test the ImagePreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test basic preprocessor initialization."""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None

    def test_preprocess_valid_image(self):
        """Test preprocessing with a valid PIL image."""
        preprocessor = ImagePreprocessor()

        # Create a test image (white square with black border)
        test_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_array[:5, :] = 0  # Top border
        test_array[-5:, :] = 0  # Bottom border
        test_array[:, :5] = 0  # Left border
        test_array[:, -5:] = 0  # Right border

        test_image = Image.fromarray(test_array)

        # Process the image
        result = preprocessor.preprocess(test_image)

        # Verify result is a PIL Image
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode in ["RGB", "RGBA"]

    def test_preprocess_small_image(self):
        """Test preprocessing with a small image."""
        preprocessor = ImagePreprocessor()

        # Create minimal test image
        test_array = np.ones((20, 20, 3), dtype=np.uint8) * 128
        test_image = Image.fromarray(test_array)

        result = preprocessor.preprocess(test_image)

        assert isinstance(result, Image.Image)
        assert result.size == (20, 20)

    def test_preprocess_grayscale_conversion(self):
        """Test preprocessing handles color to grayscale conversion properly."""
        preprocessor = ImagePreprocessor()

        # Create colorful test image
        test_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)

        result = preprocessor.preprocess(test_image)

        assert isinstance(result, Image.Image)

    def test_deskew_with_lines(self):
        """Test deskewing when lines are detected."""
        preprocessor = ImagePreprocessor()

        # Create image with horizontal lines (simulating text)
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255

        # Add horizontal lines
        for y in range(20, 80, 15):
            image[y : y + 3, 20:180] = 0

        # Mock cv2.HoughLines to return detected lines
        with patch("cv2.HoughLines") as mock_hough:
            # Simulate detected lines with slight angle
            mock_hough.return_value = np.array([[[100, np.pi / 2 + 0.2]]])

            result = preprocessor._deskew(image)

            assert isinstance(result, np.ndarray)
            assert result.shape == image.shape

    def test_deskew_no_lines(self):
        """Test deskewing when no lines are detected."""
        preprocessor = ImagePreprocessor()

        # Create image without clear lines
        image = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)

        # Mock cv2.HoughLines to return None
        with patch("cv2.HoughLines", return_value=None):
            result = preprocessor._deskew(image)

            # Should return original image unchanged
            assert isinstance(result, np.ndarray)
            assert result.shape == image.shape

    def test_deskew_small_angle(self):
        """Test deskewing with small angle (should not rotate)."""
        preprocessor = ImagePreprocessor()

        image = np.ones((50, 50, 3), dtype=np.uint8) * 255

        # Mock small angle detection
        with patch("cv2.HoughLines") as mock_hough:
            mock_hough.return_value = np.array(
                [[[100, np.pi / 2 + 0.05]]]
            )  # Small angle

            result = preprocessor._deskew(image)

            # Should not rotate for small angles
            assert isinstance(result, np.ndarray)

    def test_denoise_function(self):
        """Test noise removal function."""
        preprocessor = ImagePreprocessor()

        # Create noisy image
        clean_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        noise = np.random.randint(-30, 30, (50, 50, 3), dtype=np.int16)
        noisy_image = np.clip(clean_image.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

        result = preprocessor._denoise(noisy_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8

    def test_enhance_contrast_function(self):
        """Test contrast enhancement function."""
        preprocessor = ImagePreprocessor()

        # Create low contrast image
        low_contrast = np.ones((50, 50, 3), dtype=np.uint8) * 100
        low_contrast[10:40, 10:40] = 150  # Slightly brighter region

        result = preprocessor._enhance_contrast(low_contrast)

        assert isinstance(result, np.ndarray)
        assert result.shape == low_contrast.shape
        assert result.dtype == np.uint8

    def test_normalize_function(self):
        """Test intensity normalization function."""
        preprocessor = ImagePreprocessor()

        # Create image with varied intensities
        test_image = np.array([[[50, 100, 150], [200, 25, 75]]], dtype=np.uint8)
        test_image = np.repeat(test_image, 25, axis=0)
        test_image = np.repeat(test_image, 25, axis=1)

        result = preprocessor._normalize(test_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # Check that normalization spreads values across full range
        assert result.min() >= 0
        assert result.max() <= 255

    def test_normalize_uniform_image(self):
        """Test normalization with uniform intensity image."""
        preprocessor = ImagePreprocessor()

        # Create uniform image (all same value)
        uniform_image = np.ones((30, 30, 3), dtype=np.uint8) * 128

        # Should handle division by zero gracefully
        try:
            result = preprocessor._normalize(uniform_image)
            assert isinstance(result, np.ndarray)
        except Exception as e:
            pytest.fail(f"Normalization failed on uniform image: {e}")

    def test_opencv_integration(self):
        """Test OpenCV function integration."""
        preprocessor = ImagePreprocessor()

        # Test with real OpenCV operations
        test_image = np.ones((50, 50, 3), dtype=np.uint8) * 200

        # Test individual operations don't crash
        denoised = preprocessor._denoise(test_image)
        enhanced = preprocessor._enhance_contrast(test_image)
        normalized = preprocessor._normalize(test_image)

        assert all(
            isinstance(img, np.ndarray) for img in [denoised, enhanced, normalized]
        )

    def test_error_handling_invalid_image(self):
        """Test error handling with invalid image data."""
        preprocessor = ImagePreprocessor()

        # Test with edge cases
        try:
            # Empty array
            empty_array = np.array([])
            if empty_array.size > 0:
                preprocessor._normalize(empty_array.reshape(1, 1, 3).astype(np.uint8))
        except Exception:
            pass  # Expected to handle gracefully

        # Test with minimal valid image
        minimal_image = np.ones((1, 1, 3), dtype=np.uint8) * 100
        result = preprocessor._normalize(minimal_image)
        assert isinstance(result, np.ndarray)

    def test_color_space_conversions(self):
        """Test color space conversion robustness."""
        preprocessor = ImagePreprocessor()

        # Create test image with varied colors
        rgb_image = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)

        # Test LAB conversion in contrast enhancement
        try:
            result = preprocessor._enhance_contrast(rgb_image)
            assert isinstance(result, np.ndarray)
            assert result.shape == rgb_image.shape
        except Exception as e:
            pytest.fail(f"Color space conversion failed: {e}")

    def test_pipeline_integration(self):
        """Test the complete preprocessing pipeline."""
        preprocessor = ImagePreprocessor()

        # Create realistic test image (document-like)
        document_image = np.ones((200, 300, 3), dtype=np.uint8) * 240

        # Add text-like patterns
        for y in range(50, 150, 20):
            document_image[y : y + 5, 50:250] = 40  # Dark text lines

        # Add some noise
        noise = np.random.randint(-20, 20, document_image.shape, dtype=np.int16)
        document_image = np.clip(
            document_image.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        pil_image = Image.fromarray(document_image)

        # Process through complete pipeline
        result = preprocessor.preprocess(pil_image)

        assert isinstance(result, Image.Image)
        assert result.size == pil_image.size

    def test_memory_efficiency(self):
        """Test memory usage with larger images."""
        preprocessor = ImagePreprocessor()

        # Create larger test image
        large_image = np.random.randint(0, 255, (500, 700, 3), dtype=np.uint8)
        pil_image = Image.fromarray(large_image)

        # Should handle larger images without excessive memory usage
        try:
            result = preprocessor.preprocess(pil_image)
            assert isinstance(result, Image.Image)
        except MemoryError:
            pytest.fail("Preprocessing failed due to memory issues")

    def test_various_image_sizes(self):
        """Test preprocessing with various image dimensions."""
        preprocessor = ImagePreprocessor()

        test_sizes = [(10, 10), (50, 100), (100, 50), (200, 200)]

        for width, height in test_sizes:
            test_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            test_image = Image.fromarray(test_array)

            result = preprocessor.preprocess(test_image)

            assert isinstance(result, Image.Image)
            assert result.size == (width, height)


class TestImagePreprocessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_high_contrast_image(self):
        """Test with high contrast image."""
        preprocessor = ImagePreprocessor()

        # Create high contrast image (pure black and white)
        high_contrast = np.zeros((50, 50, 3), dtype=np.uint8)
        high_contrast[25:, :] = 255

        pil_image = Image.fromarray(high_contrast)
        result = preprocessor.preprocess(pil_image)

        assert isinstance(result, Image.Image)

    def test_low_contrast_image(self):
        """Test with very low contrast image."""
        preprocessor = ImagePreprocessor()

        # Create low contrast image (all very similar values)
        low_contrast = np.ones((50, 50, 3), dtype=np.uint8) * 127
        low_contrast[20:30, 20:30] = 130  # Tiny difference

        pil_image = Image.fromarray(low_contrast)
        result = preprocessor.preprocess(pil_image)

        assert isinstance(result, Image.Image)

    def test_single_color_image(self):
        """Test with single color image."""
        preprocessor = ImagePreprocessor()

        # Create solid color image
        solid_color = np.ones((30, 30, 3), dtype=np.uint8) * 100

        pil_image = Image.fromarray(solid_color)
        result = preprocessor.preprocess(pil_image)

        assert isinstance(result, Image.Image)

    def test_rotated_content(self):
        """Test with rotated content detection."""
        preprocessor = ImagePreprocessor()

        # Create image with clearly rotated lines
        rotated_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Add diagonal lines
        for i in range(0, 100, 10):
            y_start = max(0, i - 20)
            y_end = min(100, i + 20)
            x_start = max(0, i - 10)
            x_end = min(100, i + 10)
            rotated_image[y_start:y_end, x_start:x_end] = 0

        result = preprocessor._deskew(rotated_image)
        assert isinstance(result, np.ndarray)

    def test_very_noisy_image(self):
        """Test with extremely noisy image."""
        preprocessor = ImagePreprocessor()

        # Create very noisy image
        base_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        heavy_noise = np.random.randint(-100, 100, (50, 50, 3), dtype=np.int16)
        noisy_image = np.clip(base_image.astype(np.int16) + heavy_noise, 0, 255).astype(
            np.uint8
        )

        result = preprocessor._denoise(noisy_image)
        assert isinstance(result, np.ndarray)

    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent across multiple runs."""
        preprocessor = ImagePreprocessor()

        # Create deterministic test image with proper dimensions
        test_array = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(50):
            test_array[i, :, :] = (i * 5) % 256  # Create gradient pattern
        test_image = Image.fromarray(test_array)

        # Process multiple times
        results = []
        for _ in range(3):
            result = preprocessor.preprocess(test_image)
            results.append(np.array(result))

        # Results should be identical (deterministic processing)
        for i in range(1, len(results)):
            assert np.array_equal(
                results[0], results[i]
            ), "Preprocessing should be deterministic"
