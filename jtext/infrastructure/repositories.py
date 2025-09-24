"""
Repository Implementations.

Concrete implementations of domain repository interfaces.
"""

from typing import Dict, List, Optional
from collections import defaultdict

from ..core import Result, Ok, Err
from ..domain import (
    Document,
    ProcessingResult,
    VisionAnalysis,
    DocumentId,
    VisionAnalysisId,
    DocumentRepository,
    ProcessingResultRepository,
    VisionAnalysisRepository,
)
from .logging import get_logger


class InMemoryDocumentRepository(DocumentRepository):
    """In-memory document repository implementation."""

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self.logger = get_logger("InMemoryDocumentRepository")

    def save(self, document: Document) -> Result[None, str]:
        """Save document."""
        try:
            self._documents[document.id.value] = document
            self.logger.info(f"Document saved: {document.id.value}")
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to save document: {str(e)}")

    def find_by_id(self, document_id: DocumentId) -> Result[Optional[Document], str]:
        """Find document by ID."""
        try:
            document = self._documents.get(document_id.value)
            return Ok(document)
        except Exception as e:
            return Err(f"Failed to find document: {str(e)}")

    def find_all(self) -> Result[List[Document], str]:
        """Find all documents."""
        try:
            documents = list(self._documents.values())
            return Ok(documents)
        except Exception as e:
            return Err(f"Failed to find all documents: {str(e)}")

    def delete(self, document_id: DocumentId) -> Result[None, str]:
        """Delete document."""
        try:
            if document_id.value in self._documents:
                del self._documents[document_id.value]
                self.logger.info(f"Document deleted: {document_id.value}")
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to delete document: {str(e)}")


class InMemoryProcessingResultRepository(ProcessingResultRepository):
    """In-memory processing result repository implementation."""

    def __init__(self):
        self._results: Dict[str, ProcessingResult] = {}
        self._document_results: Dict[str, List[ProcessingResult]] = defaultdict(list)
        self.logger = get_logger("InMemoryProcessingResultRepository")

    def save(self, result: ProcessingResult) -> Result[None, str]:
        """Save processing result."""
        try:
            self._results[result.id] = result
            self._document_results[result.document_id.value].append(result)
            self.logger.info(f"Processing result saved: {result.id}")
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to save processing result: {str(e)}")

    def find_by_document_id(
        self, document_id: DocumentId
    ) -> Result[List[ProcessingResult], str]:
        """Find results by document ID."""
        try:
            results = self._document_results.get(document_id.value, [])
            return Ok(results)
        except Exception as e:
            return Err(f"Failed to find results by document ID: {str(e)}")

    def find_by_id(self, result_id: str) -> Result[Optional[ProcessingResult], str]:
        """Find result by ID."""
        try:
            result = self._results.get(result_id)
            return Ok(result)
        except Exception as e:
            return Err(f"Failed to find result by ID: {str(e)}")


class InMemoryVisionAnalysisRepository(VisionAnalysisRepository):
    """In-memory vision analysis repository implementation."""

    def __init__(self):
        self._analyses: Dict[str, VisionAnalysis] = {}
        self._document_analyses: Dict[str, List[VisionAnalysis]] = defaultdict(list)
        self.logger = get_logger("InMemoryVisionAnalysisRepository")

    def save(self, analysis: VisionAnalysis) -> Result[None, str]:
        """Save vision analysis."""
        try:
            self._analyses[analysis.id.value] = analysis
            self._document_analyses[analysis.document_id.value].append(analysis)
            self.logger.info(f"Vision analysis saved: {analysis.id.value}")
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to save vision analysis: {str(e)}")

    def find_by_id(
        self, analysis_id: VisionAnalysisId
    ) -> Result[Optional[VisionAnalysis], str]:
        """Find vision analysis by ID."""
        try:
            analysis = self._analyses.get(analysis_id.value)
            return Ok(analysis)
        except Exception as e:
            return Err(f"Failed to find vision analysis by ID: {str(e)}")

    def find_by_document_id(
        self, document_id: DocumentId
    ) -> Result[List[VisionAnalysis], str]:
        """Find analyses by document ID."""
        try:
            analyses = self._document_analyses.get(document_id.value, [])
            return Ok(analyses)
        except Exception as e:
            return Err(f"Failed to find analyses by document ID: {str(e)}")

    def find_all(self) -> Result[List[VisionAnalysis], str]:
        """Find all vision analyses."""
        try:
            analyses = list(self._analyses.values())
            return Ok(analyses)
        except Exception as e:
            return Err(f"Failed to find all vision analyses: {str(e)}")

    def delete(self, analysis_id: VisionAnalysisId) -> Result[None, str]:
        """Delete vision analysis."""
        try:
            if analysis_id.value in self._analyses:
                analysis = self._analyses[analysis_id.value]
                # Remove from document analyses
                if analysis.document_id.value in self._document_analyses:
                    self._document_analyses[analysis.document_id.value] = [
                        a
                        for a in self._document_analyses[analysis.document_id.value]
                        if a.id.value != analysis_id.value
                    ]
                del self._analyses[analysis_id.value]
                self.logger.info(f"Vision analysis deleted: {analysis_id.value}")
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to delete vision analysis: {str(e)}")
