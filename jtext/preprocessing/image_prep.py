"""
Image preprocessing utilities for OCR optimization.

This module provides image preprocessing functions to improve
OCR accuracy by normalizing, deskewing, and enhancing images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

from ..utils.logging import get_logger


logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing for OCR optimization.

    Provides methods to enhance image quality for better OCR results
    including deskewing, noise removal, and contrast adjustment.
    """

    def __init__(self):
        """Initialize the image preprocessor."""
        logger.debug("Initialized ImagePreprocessor")

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Apply comprehensive preprocessing to an image.

        Args:
            image: PIL Image object to preprocess

        Returns:
            Preprocessed PIL Image object
        """
        logger.debug("Starting image preprocessing")

        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply preprocessing steps
        processed = self._deskew(cv_image)
        processed = self._denoise(processed)
        processed = self._enhance_contrast(processed)
        processed = self._normalize(processed)

        # Convert back to PIL format
        result = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        logger.debug("Image preprocessing completed")
        return result

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in the image.

        Args:
            image: OpenCV image array

        Returns:
            Deskewed image array
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta - np.pi / 2
                angles.append(angle)

            if angles:
                avg_angle = np.median(angles)

                # Only correct if angle is significant
                if abs(avg_angle) > 0.1:  # ~6 degrees
                    # Rotate image to correct skew
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(
                        center, np.degrees(avg_angle), 1.0
                    )
                    image = cv2.warpAffine(
                        image,
                        rotation_matrix,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image.

        Args:
            image: OpenCV image array

        Returns:
            Denoised image array
        """
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: OpenCV image array

        Returns:
            Contrast-enhanced image array
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity values.

        Args:
            image: OpenCV image array

        Returns:
            Normalized image array
        """
        # Convert to float32 for processing
        normalized = image.astype(np.float32)

        # Handle uniform images (avoid division by zero)
        min_val = normalized.min()
        max_val = normalized.max()

        if max_val == min_val:
            # Uniform image - return as is
            return image

        # Normalize to 0-1 range
        normalized = (normalized - min_val) / (max_val - min_val)

        # Convert back to uint8
        normalized = (normalized * 255).astype(np.uint8)

        return normalized
