"""
Repository Interfaces.

Repository interfaces define contracts for data access without implementation details.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..core import Result
from .entities import Document, ProcessingResult, VisionAnalysis
from .value_objects import DocumentId, VisionAnalysisId


class DocumentRepository(ABC):
    """Document repository interface."""

    @abstractmethod
    def save(self, document: Document) -> Result[None, str]:
        """Save document."""
        pass

    @abstractmethod
    def find_by_id(self, document_id: DocumentId) -> Result[Optional[Document], str]:
        """Find document by ID."""
        pass

    @abstractmethod
    def find_all(self) -> Result[List[Document], str]:
        """Find all documents."""
        pass

    @abstractmethod
    def delete(self, document_id: DocumentId) -> Result[None, str]:
        """Delete document."""
        pass


class ProcessingResultRepository(ABC):
    """Processing result repository interface."""

    @abstractmethod
    def save(self, result: ProcessingResult) -> Result[None, str]:
        """Save processing result."""
        pass

    @abstractmethod
    def find_by_document_id(
        self, document_id: DocumentId
    ) -> Result[List[ProcessingResult], str]:
        """Find results by document ID."""
        pass

    @abstractmethod
    def find_by_id(self, result_id: str) -> Result[Optional[ProcessingResult], str]:
        """Find result by ID."""
        pass


class VisionAnalysisRepository(ABC):
    """Vision analysis repository interface."""

    @abstractmethod
    def save(self, analysis: VisionAnalysis) -> Result[None, str]:
        """Save vision analysis."""
        pass

    @abstractmethod
    def find_by_id(
        self, analysis_id: VisionAnalysisId
    ) -> Result[Optional[VisionAnalysis], str]:
        """Find vision analysis by ID."""
        pass

    @abstractmethod
    def find_by_document_id(
        self, document_id: DocumentId
    ) -> Result[List[VisionAnalysis], str]:
        """Find analyses by document ID."""
        pass

    @abstractmethod
    def find_all(self) -> Result[List[VisionAnalysis], str]:
        """Find all vision analyses."""
        pass

    @abstractmethod
    def delete(self, analysis_id: VisionAnalysisId) -> Result[None, str]:
        """Delete vision analysis."""
        pass
