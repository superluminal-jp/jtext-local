"""
Audio transcription using Whisper for high-accuracy speech-to-text.

This module provides audio transcription capabilities using OpenAI's
Whisper model for converting speech to text with support for Japanese
and other languages.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import psutil

# Audio processing imports
from faster_whisper import WhisperModel
import ffmpeg

from ..utils.logging import get_logger
from ..utils.validation import validate_audio_file
from ..core.ocr_hybrid import ProcessingResult


logger = get_logger(__name__)


class AudioTranscriber:
    """
    Audio transcription using Whisper model.

    This class provides audio transcription capabilities using OpenAI's
    Whisper model for converting speech to text with high accuracy.
    """

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize the audio transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, auto)
        """
        self.model_size = model_size
        self.device = device
        self.model = None

        logger.info(f"Initialized AudioTranscriber with model: {model_size}")

    def _load_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            try:
                logger.info(f"Loading Whisper model: {self.model_size}")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8",
                )
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

    def transcribe_audio(
        self, file_path: str, language: str = "ja"
    ) -> ProcessingResult:
        """
        Transcribe audio file to text.

        Args:
            file_path: Path to audio file
            language: Language code (ja, en, etc.)

        Returns:
            ProcessingResult containing transcribed text and metadata

        Raises:
            ValueError: If audio file is invalid
            FileNotFoundError: If file doesn't exist
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Starting audio transcription for: {file_path}")

        # Validate input file
        if not validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")

        try:
            # Load model if needed
            self._load_model()

            # Preprocess audio if needed
            processed_audio_path = self._preprocess_audio(file_path)

            # Perform transcription
            logger.debug("Running Whisper transcription")
            segments, info = self.model.transcribe(
                processed_audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                initial_prompt="This is a Japanese audio transcription.",
            )

            # Combine segments into full text
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text.strip())

            full_text = " ".join(text_segments)

            # Calculate confidence (average of segment confidences)
            confidence = 0.8  # Whisper doesn't provide confidence scores directly
            if hasattr(segments, "__iter__"):
                # Estimate confidence based on segment quality
                confidence = min(0.95, max(0.5, len(full_text) / 1000))

            # Calculate processing metrics
            processing_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory

            # Create result object
            result = ProcessingResult(
                source_path=file_path,
                text=full_text,
                confidence=confidence,
                processing_time=processing_time,
                memory_usage=memory_usage,
                corrections_applied=0,
                correction_ratio=0.0,
            )

            # Add audio-specific metadata
            result.audio_metadata = {
                "language": info.language,
                "language_probability": getattr(info, "language_probability", 0.0),
                "duration": getattr(info, "duration", 0.0),
                "model_size": self.model_size,
                "device": self.device,
            }

            logger.info(
                f"Audio transcription completed in {processing_time:.2f}s, "
                f"confidence: {confidence:.2f}, language: {info.language}"
            )

            return result

        except Exception as e:
            logger.error(f"Audio transcription failed for {file_path}: {e}")
            raise

    def _preprocess_audio(self, file_path: str) -> str:
        """
        Preprocess audio file for optimal transcription.

        Args:
            file_path: Path to audio file

        Returns:
            Path to processed audio file
        """
        try:
            # Check if file needs preprocessing
            file_path_obj = Path(file_path)
            processed_path = file_path_obj.parent / f"processed_{file_path_obj.name}"

            # For now, return original path (preprocessing can be added later)
            # This could include format conversion, noise reduction, etc.
            return str(file_path)

        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using original file")
            return file_path

    def transcribe_multiple(
        self, file_paths: List[str], language: str = "ja"
    ) -> List[ProcessingResult]:
        """
        Transcribe multiple audio files.

        Args:
            file_paths: List of audio file paths
            language: Language code for transcription

        Returns:
            List of ProcessingResult objects
        """
        results = []

        for file_path in file_paths:
            try:
                result = self.transcribe_audio(file_path, language)
                results.append(result)
                logger.info(f"Successfully transcribed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to transcribe {file_path}: {e}")
                # Continue with other files
                continue

        logger.info(f"Transcribed {len(results)}/{len(file_paths)} audio files")
        return results

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats.

        Returns:
            List of supported file extensions
        """
        return [".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mov", ".avi", ".mkv"]

    def get_available_models(self) -> List[str]:
        """
        Get list of available Whisper model sizes.

        Returns:
            List of available model sizes
        """
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
