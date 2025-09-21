"""
Hybrid OCR processing with Tesseract and optional LLM correction.

This module implements the core OCR functionality using Tesseract
with optional LLM-based correction for improved accuracy.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
import pytesseract
from PIL import Image
import psutil

from ..utils.logging import get_logger
from ..preprocessing.image_prep import ImagePreprocessor
from ..correction.ocr_corrector import OCRCorrector
from ..utils.validation import validate_image_file


logger = get_logger(__name__)


class ProcessingResult:
    """Container for OCR processing results with metadata."""

    def __init__(
        self,
        source_path: str,
        text: str,
        confidence: float,
        processing_time: float,
        memory_usage: float,
        corrections_applied: int = 0,
        correction_ratio: float = 0.0,
    ):
        self.source_path = source_path
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.memory_usage = memory_usage
        self.corrections_applied = corrections_applied
        self.correction_ratio = correction_ratio
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "source": str(Path(self.source_path).absolute()),
            "type": "image",
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
            ),
            "processing": {
                "pipeline": (
                    ["tesseract", "llm_correction"]
                    if self.corrections_applied > 0
                    else ["tesseract"]
                ),
                "ocr_engine": "tesseract-5.3.3",
                "llm_model": "gpt-oss" if self.corrections_applied > 0 else None,
                "confidence": {
                    "ocr_raw": self.confidence,
                    "llm_corrected": (
                        self.confidence if self.corrections_applied > 0 else None
                    ),
                },
            },
            "correction_stats": {
                "characters_changed": self.corrections_applied,
                "words_changed": self.corrections_applied // 2,  # Rough estimate
                "correction_ratio": self.correction_ratio,
                "correction_types": (
                    ["kanji_fix", "punctuation", "layout"]
                    if self.corrections_applied > 0
                    else []
                ),
            },
            "quality_metrics": {
                "character_count": len(self.text),
                "word_count": len(self.text.split()),
                "processing_time_sec": self.processing_time,
                "memory_usage_mb": self.memory_usage,
            },
        }


class HybridOCR:
    """
    Tesseract + LLM correction for high-precision OCR processing.

    This class provides the main OCR functionality with optional
    LLM-based correction for improved accuracy on Japanese text.
    """

    def __init__(
        self, llm_model: Optional[str] = None, enable_correction: bool = False
    ):
        """
        Initialize the hybrid OCR processor.

        Args:
            llm_model: LLM model name for correction (e.g., 'gpt-oss')
            enable_correction: Whether to enable LLM correction
        """
        self.enable_correction = enable_correction
        self.llm_model = llm_model
        self.preprocessor = ImagePreprocessor()
        self.corrector = OCRCorrector(model=llm_model) if enable_correction else None

        logger.info(f"Initialized HybridOCR with correction: {enable_correction}")

    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Process a single image through the OCR pipeline.

        Args:
            image_path: Path to the image file

        Returns:
            ProcessingResult containing extracted text and metadata

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image file is invalid
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Starting OCR processing for: {image_path}")

        # Validate input file
        if not validate_image_file(image_path):
            raise ValueError(f"Invalid image file: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocessor.preprocess(image)

            # Perform OCR with Tesseract
            logger.debug("Running Tesseract OCR")
            ocr_text = pytesseract.image_to_string(
                processed_image,
                lang="jpn+eng",
                config="--psm 6",  # Uniform block of text
            )

            # Get confidence score
            ocr_data = pytesseract.image_to_data(
                processed_image, lang="jpn+eng", output_type=pytesseract.Output.DICT
            )
            confidence = self._calculate_confidence(ocr_data)

            # Apply LLM correction if enabled
            corrected_text = ocr_text
            corrections_applied = 0
            correction_ratio = 0.0

            if self.enable_correction and self.corrector:
                logger.debug("Applying LLM correction")
                corrected_text, corrections_applied = self.corrector.correct(ocr_text)
                correction_ratio = (
                    corrections_applied / len(ocr_text) if ocr_text else 0.0
                )

            # Calculate processing metrics
            processing_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory

            result = ProcessingResult(
                source_path=image_path,
                text=corrected_text,
                confidence=confidence,
                processing_time=processing_time,
                memory_usage=memory_usage,
                corrections_applied=corrections_applied,
                correction_ratio=correction_ratio,
            )

            logger.info(
                f"OCR processing completed in {processing_time:.2f}s, "
                f"confidence: {confidence:.2f}, corrections: {corrections_applied}"
            )

            return result

        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            raise

    def _calculate_confidence(self, ocr_data: Dict[str, Any]) -> float:
        """
        Calculate average confidence score from Tesseract output.

        Args:
            ocr_data: Tesseract output data dictionary

        Returns:
            Average confidence score (0.0 to 1.0)
        """
        confidences = [int(conf) for conf in ocr_data["conf"] if int(conf) > 0]
        return sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
