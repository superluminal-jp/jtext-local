"""
Use Cases.

Use cases orchestrate the flow of data to and from entities and coordinate application activities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

from ..core import Result, Ok, Err, PerformanceMonitor, generate_id
from ..domain import (
    Document,
    ImageDocument,
    AudioDocument,
    ProcessingResult,
    DocumentId,
    DocumentType,
    Language,
    ProcessingStatus,
    Confidence,
    FilePath,
    DocumentRepository,
    ProcessingResultRepository,
    EventPublisher,
    DocumentProcessedEvent,
    ProcessingFailedEvent,
)
from .dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    DocumentDTO,
    ProcessingResultDTO,
    ProcessingStatisticsDTO,
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    VisionAnalysisDTO,
)


class UseCase(ABC):
    """Base use case interface."""

    def __init__(self, event_publisher: EventPublisher):
        self.event_publisher = event_publisher

    @abstractmethod
    def execute(self, request: Any) -> Result[Any, str]:
        """Execute use case."""
        pass


class ProcessDocumentUseCase(UseCase):
    """Use case for processing documents."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_result_repository: ProcessingResultRepository,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.document_repository = document_repository
        self.processing_result_repository = processing_result_repository

    def execute(
        self, request: ProcessDocumentRequest
    ) -> Result[ProcessDocumentResponse, str]:
        """Execute document processing."""
        with PerformanceMonitor("process_document") as monitor:
            try:
                # Create document entity
                document = self._create_document(request)

                # Save document
                save_result = self.document_repository.save(document)
                if save_result.is_err:
                    return Err(f"Failed to save document: {save_result.unwrap_err()}")

                # Process document based on type
                if request.document_type == DocumentType.IMAGE:
                    return self._process_image_document(document, request)
                elif request.document_type == DocumentType.AUDIO:
                    return self._process_audio_document(document, request)
                else:
                    return Err(f"Unsupported document type: {request.document_type}")

            except Exception as e:
                return Err(f"Document processing failed: {str(e)}")

    def _create_document(self, request: ProcessDocumentRequest) -> Document:
        """Create document entity from request."""
        document_id = DocumentId.generate()
        file_path = FilePath(request.file_path)

        if request.document_type == DocumentType.IMAGE:
            return ImageDocument(
                id=document_id,
                file_path=file_path,
                document_type=request.document_type,
                language=request.language,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        elif request.document_type == DocumentType.AUDIO:
            return AudioDocument(
                id=document_id,
                file_path=file_path,
                document_type=request.document_type,
                language=request.language,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        else:
            return Document(
                id=document_id,
                file_path=file_path,
                document_type=request.document_type,
                language=request.language,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def _process_image_document(
        self, document: Document, request: ProcessDocumentRequest
    ) -> Result[ProcessDocumentResponse, str]:
        """Process image document."""
        # Update status to processing
        document.update_status(ProcessingStatus.PROCESSING)
        self.document_repository.save(document)

        # Simulate processing (in real implementation, this would call OCR service)
        processing_results = []

        # Create mock OCR result
        ocr_result = ProcessingResult(
            id=generate_id(),
            document_id=document.id,
            result_type="OCR",
            content="Extracted text from image",
            confidence=Confidence.from_float(0.95),
            processing_time=2.5,
            created_at=datetime.now(),
        )

        # Save result
        save_result = self.processing_result_repository.save(ocr_result)
        if save_result.is_err:
            return Err(f"Failed to save OCR result: {save_result.unwrap_err()}")

        processing_results.append(ocr_result.to_dict())

        # Update document status
        document.update_status(ProcessingStatus.COMPLETED)
        self.document_repository.save(document)

        # Publish event
        event = DocumentProcessedEvent(
            document_id=document.id.value,
            processing_result=processing_results[0],
            processing_time=2.5,
        )
        self.event_publisher.publish(event)

        return Ok(
            ProcessDocumentResponse(
                document_id=document.id.value,
                status=ProcessingStatus.COMPLETED,
                processing_time=2.5,
                results=processing_results,
                metadata={},
            )
        )

    def _process_audio_document(
        self, document: Document, request: ProcessDocumentRequest
    ) -> Result[ProcessDocumentResponse, str]:
        """Process audio document."""
        # Update status to processing
        document.update_status(ProcessingStatus.PROCESSING)
        self.document_repository.save(document)

        # Simulate processing (in real implementation, this would call transcription service)
        processing_results = []

        # Create mock transcription result
        transcription_result = ProcessingResult(
            id=generate_id(),
            document_id=document.id,
            result_type="TRANSCRIPTION",
            content="Transcribed audio to text",
            confidence=Confidence.from_float(0.88),
            processing_time=5.2,
            created_at=datetime.now(),
        )

        # Save result
        save_result = self.processing_result_repository.save(transcription_result)
        if save_result.is_err:
            return Err(
                f"Failed to save transcription result: {save_result.unwrap_err()}"
            )

        processing_results.append(transcription_result.to_dict())

        # Update document status
        document.update_status(ProcessingStatus.COMPLETED)
        self.document_repository.save(document)

        # Publish event
        event = DocumentProcessedEvent(
            document_id=document.id.value,
            processing_result=processing_results[0],
            processing_time=5.2,
        )
        self.event_publisher.publish(event)

        return Ok(
            ProcessDocumentResponse(
                document_id=document.id.value,
                status=ProcessingStatus.COMPLETED,
                processing_time=5.2,
                results=processing_results,
                metadata={},
            )
        )


class ProcessImageUseCase(UseCase):
    """Use case for processing images."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_result_repository: ProcessingResultRepository,
        ocr_service,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.document_repository = document_repository
        self.processing_result_repository = processing_result_repository
        self.ocr_service = ocr_service

    def execute(
        self, request: ProcessDocumentRequest
    ) -> Result[ProcessDocumentResponse, str]:
        """Execute image processing."""
        # Implementation would use OCR service
        return Err("Not implemented yet")


class ProcessAudioUseCase(UseCase):
    """Use case for processing audio."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_result_repository: ProcessingResultRepository,
        transcription_service,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.document_repository = document_repository
        self.processing_result_repository = processing_result_repository
        self.transcription_service = transcription_service

    def execute(
        self, request: ProcessDocumentRequest
    ) -> Result[ProcessDocumentResponse, str]:
        """Execute audio processing."""
        # Implementation would use transcription service
        return Err("Not implemented yet")


class ProcessStructuredDocumentUseCase(UseCase):
    """Use case for processing structured documents."""

    def execute(
        self, request: ProcessDocumentRequest
    ) -> Result[ProcessDocumentResponse, str]:
        """Execute structured document processing."""
        return Err("Not implemented yet")


class GetProcessingResultUseCase(UseCase):
    """Use case for getting processing results."""

    def __init__(
        self,
        processing_result_repository: ProcessingResultRepository,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.processing_result_repository = processing_result_repository

    def execute(self, document_id: str) -> Result[List[ProcessingResultDTO], str]:
        """Execute get processing results."""
        try:
            doc_id = DocumentId(document_id)
            result = self.processing_result_repository.find_by_document_id(doc_id)

            if result.is_err:
                return Err(f"Failed to find results: {result.unwrap_err()}")

            results = result.unwrap()
            dtos = [
                ProcessingResultDTO(
                    id=r.id,
                    document_id=r.document_id.value,
                    result_type=r.result_type,
                    content=r.content,
                    confidence=r.confidence.to_float(),
                    processing_time=r.processing_time,
                    created_at=r.created_at.isoformat(),
                    metadata=r.metadata,
                )
                for r in results
            ]

            return Ok(dtos)

        except Exception as e:
            return Err(f"Failed to get processing results: {str(e)}")


class ListDocumentsUseCase(UseCase):
    """Use case for listing documents."""

    def __init__(
        self, document_repository: DocumentRepository, event_publisher: EventPublisher
    ):
        super().__init__(event_publisher)
        self.document_repository = document_repository

    def execute(self) -> Result[List[DocumentDTO], str]:
        """Execute list documents."""
        try:
            result = self.document_repository.find_all()

            if result.is_err:
                return Err(f"Failed to find documents: {result.unwrap_err()}")

            documents = result.unwrap()
            dtos = [
                DocumentDTO(
                    id=d.id.value,
                    file_path=d.file_path.value,
                    document_type=d.document_type.value,
                    language=d.language.value,
                    status=d.status.value,
                    created_at=d.created_at.isoformat(),
                    updated_at=d.updated_at.isoformat(),
                    metadata=d.metadata,
                )
                for d in documents
            ]

            return Ok(dtos)

        except Exception as e:
            return Err(f"Failed to list documents: {str(e)}")


class GetProcessingStatisticsUseCase(UseCase):
    """Use case for getting processing statistics."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_result_repository: ProcessingResultRepository,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.document_repository = document_repository
        self.processing_result_repository = processing_result_repository

    def execute(self) -> Result[ProcessingStatisticsDTO, str]:
        """Execute get processing statistics."""
        try:
            # Get all documents
            documents_result = self.document_repository.find_all()
            if documents_result.is_err:
                return Err(f"Failed to get documents: {documents_result.unwrap_err()}")

            documents = documents_result.unwrap()

            # Calculate statistics
            total_documents = len(documents)
            completed_documents = len(
                [d for d in documents if d.status == ProcessingStatus.COMPLETED]
            )
            failed_documents = len(
                [d for d in documents if d.status == ProcessingStatus.FAILED]
            )

            # Calculate success rate
            success_rate = (
                (completed_documents / total_documents * 100)
                if total_documents > 0
                else 0
            )

            # Group by type and status
            documents_by_type = {}
            documents_by_status = {}

            for doc in documents:
                doc_type = doc.document_type.value
                doc_status = doc.status.value

                documents_by_type[doc_type] = documents_by_type.get(doc_type, 0) + 1
                documents_by_status[doc_status] = (
                    documents_by_status.get(doc_status, 0) + 1
                )

            # Calculate average processing time (mock)
            average_processing_time = 3.5

            return Ok(
                ProcessingStatisticsDTO(
                    total_documents=total_documents,
                    completed_documents=completed_documents,
                    failed_documents=failed_documents,
                    average_processing_time=average_processing_time,
                    success_rate=success_rate,
                    documents_by_type=documents_by_type,
                    documents_by_status=documents_by_status,
                )
            )

        except Exception as e:
            return Err(f"Failed to get processing statistics: {str(e)}")


class VisionAnalysisUseCase(UseCase):
    """Use case for vision analysis."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        vision_analysis_repository,
        vision_analysis_service,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.document_repository = document_repository
        self.vision_analysis_repository = vision_analysis_repository
        self.vision_analysis_service = vision_analysis_service

    def execute(
        self, request: VisionAnalysisRequest
    ) -> Result[VisionAnalysisResponse, str]:
        """Execute vision analysis."""
        with PerformanceMonitor("vision_analysis") as monitor:
            try:
                # Find or create document
                document_result = self._find_or_create_document(request)
                if document_result.is_err:
                    return Err(
                        f"Failed to find/create document: {document_result.unwrap_err()}"
                    )

                document = document_result.unwrap()

                # Perform vision analysis
                analysis_result = self.vision_analysis_service.analyze_document(
                    image_path=request.image_path,
                    language=request.language,
                    analysis_type=request.analysis_type,
                    model_name=request.model_name,
                )

                if analysis_result.is_err:
                    return Err(
                        f"Vision analysis failed: {analysis_result.unwrap_err()}"
                    )

                vision_analysis = analysis_result.unwrap()

                # Save analysis result
                save_result = self.vision_analysis_repository.save(vision_analysis)
                if save_result.is_err:
                    return Err(
                        f"Failed to save vision analysis: {save_result.unwrap_err()}"
                    )

                # Create response
                response = VisionAnalysisResponse(
                    analysis_id=vision_analysis.id.value,
                    document_id=vision_analysis.document_id.value,
                    analysis_type=vision_analysis.analysis_type.value,
                    confidence=vision_analysis.confidence.to_float(),
                    processing_time_ms=vision_analysis.processing_time_ms,
                    model_used=vision_analysis.model_used,
                    layout_structure=vision_analysis.to_dict().get("layout_structure"),
                    text_regions=vision_analysis.to_dict().get("text_regions", []),
                    visual_elements=vision_analysis.to_dict().get(
                        "visual_elements", []
                    ),
                    quality_assessment=vision_analysis.to_dict().get(
                        "quality_assessment"
                    ),
                    content_categories=vision_analysis.content_categories,
                    metadata=vision_analysis.metadata,
                )

                # Publish event
                from ..domain import VisionAnalysisCompletedEvent

                event = VisionAnalysisCompletedEvent(
                    analysis_id=vision_analysis.id.value,
                    document_id=vision_analysis.document_id.value,
                    analysis_type=vision_analysis.analysis_type.value,
                    confidence=vision_analysis.confidence.to_float(),
                    processing_time_ms=vision_analysis.processing_time_ms,
                )
                self.event_publisher.publish(event)

                return Ok(response)

            except Exception as e:
                return Err(f"Vision analysis failed: {str(e)}")

    def _find_or_create_document(
        self, request: VisionAnalysisRequest
    ) -> Result[Document, str]:
        """Find or create document for vision analysis."""
        try:
            # For now, create a new document for each analysis
            # In a real implementation, you might want to find existing documents
            document_id = DocumentId.generate()
            file_path = FilePath(request.image_path)

            document = ImageDocument(
                id=document_id,
                file_path=file_path,
                document_type=DocumentType.IMAGE,
                language=request.language,
                status=ProcessingStatus.PROCESSING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # Save document
            save_result = self.document_repository.save(document)
            if save_result.is_err:
                return Err(f"Failed to save document: {save_result.unwrap_err()}")

            return Ok(document)

        except Exception as e:
            return Err(f"Failed to create document: {str(e)}")


class GetVisionAnalysisUseCase(UseCase):
    """Use case for getting vision analysis results."""

    def __init__(
        self,
        vision_analysis_repository,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.vision_analysis_repository = vision_analysis_repository

    def execute(self, analysis_id: str) -> Result[VisionAnalysisDTO, str]:
        """Execute get vision analysis."""
        try:
            from ..domain import VisionAnalysisId

            analysis_id_obj = VisionAnalysisId(analysis_id)

            result = self.vision_analysis_repository.find_by_id(analysis_id_obj)
            if result.is_err:
                return Err(f"Failed to find vision analysis: {result.unwrap_err()}")

            analysis = result.unwrap()
            dto = VisionAnalysisDTO(
                id=analysis.id.value,
                document_id=analysis.document_id.value,
                analysis_type=analysis.analysis_type.value,
                confidence=analysis.confidence.to_float(),
                processing_time_ms=analysis.processing_time_ms,
                model_used=analysis.model_used,
                created_at=analysis.created_at.isoformat(),
                layout_structure=analysis.to_dict().get("layout_structure"),
                text_regions=analysis.to_dict().get("text_regions", []),
                visual_elements=analysis.to_dict().get("visual_elements", []),
                quality_assessment=analysis.to_dict().get("quality_assessment"),
                content_categories=analysis.content_categories,
                metadata=analysis.metadata,
            )

            return Ok(dto)

        except Exception as e:
            return Err(f"Failed to get vision analysis: {str(e)}")


class ListVisionAnalysesUseCase(UseCase):
    """Use case for listing vision analyses."""

    def __init__(
        self,
        vision_analysis_repository,
        event_publisher: EventPublisher,
    ):
        super().__init__(event_publisher)
        self.vision_analysis_repository = vision_analysis_repository

    def execute(
        self, document_id: Optional[str] = None
    ) -> Result[List[VisionAnalysisDTO], str]:
        """Execute list vision analyses."""
        try:
            if document_id:
                from ..domain import DocumentId

                doc_id = DocumentId(document_id)
                result = self.vision_analysis_repository.find_by_document_id(doc_id)
            else:
                result = self.vision_analysis_repository.find_all()

            if result.is_err:
                return Err(f"Failed to find vision analyses: {result.unwrap_err()}")

            analyses = result.unwrap()
            dtos = [
                VisionAnalysisDTO(
                    id=analysis.id.value,
                    document_id=analysis.document_id.value,
                    analysis_type=analysis.analysis_type.value,
                    confidence=analysis.confidence.to_float(),
                    processing_time_ms=analysis.processing_time_ms,
                    model_used=analysis.model_used,
                    created_at=analysis.created_at.isoformat(),
                    layout_structure=analysis.to_dict().get("layout_structure"),
                    text_regions=analysis.to_dict().get("text_regions", []),
                    visual_elements=analysis.to_dict().get("visual_elements", []),
                    quality_assessment=analysis.to_dict().get("quality_assessment"),
                    content_categories=analysis.content_categories,
                    metadata=analysis.metadata,
                )
                for analysis in analyses
            ]

            return Ok(dtos)

        except Exception as e:
            return Err(f"Failed to list vision analyses: {str(e)}")
