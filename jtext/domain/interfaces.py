"""
Domain Service Interfaces.

Interfaces for external services that the domain depends on.
"""

from abc import ABC, abstractmethod
from typing import List

from ..core import Result
from .entities import OCRResult, TranscriptionResult
from .events import DomainEvent
from .value_objects import Language


class OCRService(ABC):
    """OCR service interface."""

    @abstractmethod
    def extract_text(
        self, image_path: str, language: Language
    ) -> Result[OCRResult, str]:
        """Extract text from image."""
        pass


class TranscriptionService(ABC):
    """Transcription service interface."""

    @abstractmethod
    def transcribe_audio(
        self, audio_path: str, language: Language
    ) -> Result[TranscriptionResult, str]:
        """Transcribe audio to text."""
        pass


class CorrectionService(ABC):
    """Text correction service interface."""

    @abstractmethod
    def correct_text(self, text: str, language: Language) -> Result[str, str]:
        """Correct text using LLM."""
        pass


class EventPublisher(ABC):
    """Event publisher interface."""

    @abstractmethod
    def publish(self, event: DomainEvent) -> Result[None, str]:
        """Publish domain event."""
        pass

    @abstractmethod
    def publish_all(self, events: List[DomainEvent]) -> Result[None, str]:
        """Publish multiple events."""
        pass
