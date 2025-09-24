"""
External Service Implementations.

Concrete implementations of domain service interfaces.
"""

from datetime import datetime
from typing import List, Callable

try:
    import pytesseract
    from PIL import Image

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from faster_whisper import WhisperModel

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

from ..core import Result, Ok, Err, generate_id
from ..domain import (
    OCRResult,
    TranscriptionResult,
    DocumentId,
    Language,
    Confidence,
    OCRService,
    TranscriptionService,
    CorrectionService,
    EventPublisher,
    DomainEvent,
)
from .errors import ExternalServiceError
from .logging import get_logger


class TesseractOCRService(OCRService):
    """Tesseract OCR service implementation."""

    def __init__(self):
        if not HAS_TESSERACT:
            raise ImportError(
                "Tesseract dependencies not available. Install with: pip install pytesseract pillow"
            )
        self.logger = get_logger("TesseractOCRService")

    def extract_text(
        self, image_path: str, language: Language
    ) -> Result[OCRResult, str]:
        """Extract text from image using Tesseract."""
        if not HAS_TESSERACT:
            return Err("Tesseract dependencies not available")

        try:
            # Configure Tesseract
            config = f"--oem 3 --psm 6 -l {language.value}"

            # Extract text
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config=config)

            # Get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Create OCR result
            result = OCRResult(
                id=generate_id(),
                document_id=DocumentId.generate(),  # This should be passed in
                result_type="OCR",
                content=text.strip(),
                confidence=Confidence.from_percentage(avg_confidence),
                processing_time=0.0,  # This should be measured
                created_at=datetime.now(),
            )

            return Ok(result)

        except Exception as e:
            error_msg = f"OCR extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return Err(error_msg)


class WhisperTranscriptionService(TranscriptionService):
    """Whisper transcription service implementation."""

    def __init__(self, model_name: str = "base"):
        if not HAS_WHISPER:
            raise ImportError(
                "Whisper dependencies not available. Install with: pip install faster-whisper"
            )
        self.model_name = model_name
        self.model = None
        self.logger = get_logger("WhisperTranscriptionService")

    def _load_model(self) -> None:
        """Load Whisper model."""
        if not HAS_WHISPER:
            return
        if self.model is None:
            self.model = WhisperModel(self.model_name)

    def transcribe_audio(
        self, audio_path: str, language: Language
    ) -> Result[TranscriptionResult, str]:
        """Transcribe audio using Whisper."""
        if not HAS_WHISPER:
            return Err("Whisper dependencies not available")

        try:
            self._load_model()

            # Transcribe audio
            segments, info = self.model.transcribe(audio_path, language=language.value)

            # Combine segments
            text = " ".join([segment.text for segment in segments])

            # Create transcription result
            result = TranscriptionResult(
                id=generate_id(),
                document_id=DocumentId.generate(),  # This should be passed in
                result_type="TRANSCRIPTION",
                content=text.strip(),
                confidence=Confidence.from_float(info.language_probability),
                processing_time=0.0,  # This should be measured
                created_at=datetime.now(),
                language_detected=info.language,
            )

            return Ok(result)

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            self.logger.error(error_msg)
            return Err(error_msg)


class OllamaCorrectionService(CorrectionService):
    """Ollama text correction service implementation."""

    def __init__(
        self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = get_logger("OllamaCorrectionService")

    def correct_text(self, text: str, language: Language) -> Result[str, str]:
        """Correct text using Ollama."""
        if not HAS_REQUESTS:
            return Err("Requests dependency not available")

        try:
            prompt = f"Please correct the following Japanese text for grammar and spelling errors:\n\n{text}\n\nCorrected text:"

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=30,
            )

            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code}"
                self.logger.error(error_msg)
                return Err(error_msg)

            result = response.json()
            corrected_text = result.get("response", text)

            return Ok(corrected_text.strip())

        except Exception as e:
            if not HAS_REQUESTS:
                error_msg = "Requests dependency not available"
            else:
                error_msg = f"Text correction failed: {str(e)}"
            self.logger.error(error_msg)
            return Err(error_msg)


class EventPublisherService(EventPublisher):
    """Event publisher service implementation."""

    def __init__(self):
        self.logger = get_logger("EventPublisherService")
        self._subscribers: List[Callable[[DomainEvent], None]] = []

    def subscribe(self, handler: Callable[[DomainEvent], None]) -> None:
        """Subscribe to domain events."""
        self._subscribers.append(handler)

    def publish(self, event: DomainEvent) -> Result[None, str]:
        """Publish domain event."""
        try:
            self.logger.info(f"Publishing event: {event.__class__.__name__}")

            # Notify subscribers
            for handler in self._subscribers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {str(e)}")

            return Ok(None)

        except Exception as e:
            return Err(f"Failed to publish event: {str(e)}")

    def publish_all(self, events: List[DomainEvent]) -> Result[None, str]:
        """Publish multiple events."""
        try:
            for event in events:
                result = self.publish(event)
                if result.is_err():
                    return result

            return Ok(None)

        except Exception as e:
            return Err(f"Failed to publish events: {str(e)}")
