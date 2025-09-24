"""
Processing Result Output Infrastructure.

Implements AGENTS.md compliant result persistence with structured output formats.
Supports JSON, TXT, and Markdown formats for different stakeholder needs.

Features:
- Multi-format output (JSON, TXT, Markdown)
- Structured metadata preservation
- File-based persistence with audit trails
- Performance metrics integration
- Error handling with recovery patterns

Architecture Decision:
- Implements Repository pattern for result persistence
- Follows Clean Architecture by separating output concerns
- Provides abstraction over file system operations

@author: jtext Development Team
@since: 1.0.0
@compliance: AGENTS.md Documentation Standards, Clean Architecture
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from ..core import Result, Ok, Err
from ..domain import ProcessingResult, Document
from .logging import get_logger, start_performance_tracking


class OutputFormat(Enum):
    """
    Supported output formats for processing results.

    Each format serves different stakeholder needs:
    - JSON: Machine-readable, API integration
    - TXT: Human-readable, simple consumption
    - MARKDOWN: Documentation, reporting
    """

    JSON = "json"
    TXT = "txt"
    MARKDOWN = "md"


@dataclass
class OutputMetadata:
    """
    Metadata for output files following AGENTS.md documentation standards.

    Provides traceability and audit information for processed results.
    """

    source_file: str
    processing_timestamp: datetime
    correlation_id: str
    processing_pipeline: List[str]
    model_versions: Dict[str, str]
    confidence_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    output_format: OutputFormat
    file_size_bytes: int
    checksum: Optional[str] = None


class ProcessingResultOutputService:
    """
    Service for outputting processing results to files.

    Implements Clean Architecture principles by providing infrastructure
    for result persistence while maintaining separation from business logic.

    Supports multiple output formats and includes comprehensive metadata
    for audit trails and traceability as required by AGENTS.md.
    """

    def __init__(self, output_directory: str = "./results"):
        """
        Initialize output service.

        Args:
            output_directory: Base directory for output files
        """
        self.output_directory = Path(output_directory)
        self.logger = get_logger("jtext.infrastructure.output")
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """
        Ensure output directory exists with proper permissions.

        Creates directory structure following security best practices.
        """
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)

            # Log directory creation for audit trail
            self.logger.info(
                "Output directory initialized",
                operation="directory_setup",
                output_directory=str(self.output_directory),
                tags={"category": "system", "component": "output"},
            )

        except Exception as e:
            self.logger.error(
                "Failed to create output directory",
                operation="directory_setup",
                error=str(e),
                output_directory=str(self.output_directory),
                tags={"category": "error", "component": "output"},
            )
            raise

    def save_processing_result(
        self,
        result: ProcessingResult,
        document: Document,
        output_format: OutputFormat = OutputFormat.JSON,
        correlation_id: Optional[str] = None,
        custom_filename: Optional[str] = None,
    ) -> Result[str, Exception]:
        """
        Save processing result to file with comprehensive metadata.

        Implements AGENTS.md standards for result persistence and audit trails.

        Args:
            result: Processing result to save
            document: Source document information
            output_format: Desired output format
            correlation_id: Correlation ID for tracing
            custom_filename: Optional custom filename

        Returns:
            Result containing output file path or error
        """
        # Start performance tracking
        metrics = start_performance_tracking()

        try:
            # Generate filename
            filename = self._generate_filename(document, output_format, custom_filename)
            output_path = self.output_directory / filename

            # Create output content based on format
            content = self._format_content(result, document, output_format)

            # Create metadata
            metadata = self._create_metadata(
                result, document, output_format, correlation_id
            )

            # Write file with atomic operation
            self._write_file_atomic(output_path, content, metadata)

            # Finish performance tracking
            metrics.finish()

            # Log successful save
            self.logger.info(
                "Processing result saved successfully",
                operation="save_result",
                output_file=str(output_path),
                output_format=output_format.value,
                file_size_bytes=len(content.encode("utf-8")),
                duration_ms=metrics.duration_ms,
                tags={"category": "business", "component": "output"},
            )

            return Ok(str(output_path))

        except Exception as e:
            # Log error with context
            self.logger.error(
                "Failed to save processing result",
                operation="save_result",
                error=str(e),
                output_format=output_format.value,
                document_id=str(result.document_id.value),
                tags={"category": "error", "component": "output"},
            )

            return Err(e)

    def _generate_filename(
        self,
        document: Document,
        output_format: OutputFormat,
        custom_filename: Optional[str] = None,
    ) -> str:
        """
        Generate output filename with timestamp and format.

        Args:
            document: Source document
            output_format: Output format
            custom_filename: Optional custom filename

        Returns:
            Generated filename
        """
        if custom_filename:
            base_name = custom_filename
        else:
            # Extract base name from source file
            source_path = Path(document.file_path.value)
            base_name = source_path.stem

        # Add timestamp for uniqueness
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        return f"{base_name}_{timestamp}.{output_format.value}"

    def _format_content(
        self, result: ProcessingResult, document: Document, output_format: OutputFormat
    ) -> str:
        """
        Format processing result content based on output format.

        Args:
            result: Processing result
            document: Source document
            output_format: Desired output format

        Returns:
            Formatted content string
        """
        if output_format == OutputFormat.JSON:
            return self._format_json(result, document)
        elif output_format == OutputFormat.TXT:
            return self._format_txt(result, document)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(result, document)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _format_json(self, result: ProcessingResult, document: Document) -> str:
        """Format result as JSON for machine consumption."""
        output_data = {
            "document": {
                "id": result.document_id.value,
                "source_file": document.file_path.value,
                "document_type": document.document_type.value,
                "language": document.language.value,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
            },
            "processing": {
                "result_type": result.result_type.value,
                "content": result.content,
                "confidence": result.confidence.to_float(),
                "processing_time_seconds": result.processing_time,
                "created_at": result.created_at.isoformat(),
            },
            "metadata": result.metadata,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
        }

        return json.dumps(output_data, indent=2, ensure_ascii=False)

    def _format_txt(self, result: ProcessingResult, document: Document) -> str:
        """Format result as plain text for human consumption."""
        lines = [
            "# jtext Processing Result",
            "",
            f"Source File: {document.file_path.value}",
            f"Document Type: {document.document_type.value}",
            f"Language: {document.language.value}",
            f"Processing Time: {result.processing_time:.2f} seconds",
            f"Confidence: {result.confidence.to_percentage():.2f}%",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Extracted Content",
            "",
            result.content,
            "",
            "## Processing Metadata",
            "",
        ]

        # Add metadata
        for key, value in result.metadata.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _format_markdown(self, result: ProcessingResult, document: Document) -> str:
        """Format result as Markdown for documentation."""
        return f"""# Processing Result Report

## Document Information

| Field | Value |
|-------|-------|
| **Source File** | `{document.file_path.value}` |
| **Document Type** | {document.document_type.value} |
| **Language** | {document.language.value} |
| **Processing Time** | {result.processing_time:.2f} seconds |
| **Confidence Score** | {result.confidence.to_percentage():.2f}% |
| **Generated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |

## Extracted Content

```
{result.content}
```

## Processing Metadata

{self._format_metadata_table(result.metadata)}

---
*Generated by jtext v1.0.0*
"""

    def _format_metadata_table(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as Markdown table."""
        if not metadata:
            return "*No additional metadata available*"

        lines = ["| Key | Value |", "|-----|-------|"]
        for key, value in metadata.items():
            lines.append(f"| {key} | {value} |")

        return "\n".join(lines)

    def _create_metadata(
        self,
        result: ProcessingResult,
        document: Document,
        output_format: OutputFormat,
        correlation_id: Optional[str],
    ) -> OutputMetadata:
        """Create comprehensive metadata for output file."""
        return OutputMetadata(
            source_file=document.file_path.value,
            processing_timestamp=datetime.now(timezone.utc),
            correlation_id=correlation_id or "unknown",
            processing_pipeline=["ocr", "correction"],  # TODO: Make dynamic
            model_versions={
                "tesseract": "5.0",
                "llm": "unknown",
            },  # TODO: Get actual versions
            confidence_scores={"overall": result.confidence.to_float()},
            performance_metrics={"processing_time_seconds": result.processing_time},
            output_format=output_format,
            file_size_bytes=0,  # Will be set after writing
        )

    def _write_file_atomic(
        self, file_path: Path, content: str, metadata: OutputMetadata
    ) -> None:
        """
        Write file atomically with metadata.

        Implements atomic write pattern to prevent partial file corruption.

        Args:
            file_path: Target file path
            content: Content to write
            metadata: File metadata
        """
        # Write to temporary file first
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

        try:
            # Write content
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Write metadata sidecar file
            metadata_path = file_path.with_suffix(file_path.suffix + ".meta.json")
            metadata.file_size_bytes = len(content.encode("utf-8"))

            with open(metadata_path.with_suffix(".tmp"), "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            # Atomic move
            temp_path.rename(file_path)
            metadata_path.with_suffix(".tmp").rename(metadata_path)

        except Exception as e:
            # Clean up temporary files on error
            temp_path.unlink(missing_ok=True)
            metadata_path.with_suffix(".tmp").unlink(missing_ok=True)
            raise

    def list_output_files(self) -> List[Dict[str, Any]]:
        """
        List all output files with metadata.

        Returns:
            List of file information dictionaries
        """
        files = []

        try:
            for file_path in self.output_directory.glob("*"):
                if file_path.suffix in [".json", ".txt", ".md"]:
                    metadata_path = file_path.with_suffix(
                        file_path.suffix + ".meta.json"
                    )

                    file_info = {
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                        "created_at": datetime.fromtimestamp(file_path.stat().st_ctime),
                        "metadata": None,
                    }

                    # Load metadata if available
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                file_info["metadata"] = json.load(f)
                        except Exception:
                            pass  # Continue without metadata

                    files.append(file_info)

        except Exception as e:
            self.logger.error(
                "Failed to list output files",
                operation="list_files",
                error=str(e),
                tags={"category": "error", "component": "output"},
            )

        return files


def get_output_service(
    output_directory: str = "./results",
) -> ProcessingResultOutputService:
    """
    Get processing result output service.

    Args:
        output_directory: Base directory for output files

    Returns:
        Configured output service instance
    """
    return ProcessingResultOutputService(output_directory)
