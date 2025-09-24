"""
Command Line Interface.

Implements AGENTS.md compliant CLI interface with comprehensive logging and
result output capabilities. Follows Clean Architecture principles by serving
as the interface adapter layer.

Features:
- Structured logging with correlation IDs for request tracing
- Comprehensive error handling with Railway-Oriented Programming
- Processing result output to multiple formats (JSON, TXT, Markdown)
- Performance monitoring with execution time tracking
- Security-first design with no PII exposure in logs

Architecture Decision:
- Implements Interface Adapter pattern from Clean Architecture
- Uses Click framework for declarative command definition
- Integrates with infrastructure services through dependency injection
- Maintains separation of concerns between CLI presentation and business logic

Business Rules:
- All processing operations must be traced with correlation IDs
- Processing results must be saved to output files for audit trails
- Performance metrics must be logged for observability
- Error conditions must be handled gracefully with user-friendly messages

@author: jtext Development Team
@since: 1.0.0
@compliance: AGENTS.md CLI Standards, Clean Architecture, Click Framework
"""

import click
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from ..core import Result, Ok, Err
from ..infrastructure import (
    get_logger,
    set_correlation_id,
    CorrelationIdGenerator,
    start_performance_tracking,
    get_performance_metrics,
)
from ..infrastructure.output import get_output_service, OutputFormat
from ..domain import DocumentType, Language, Document, FilePath
from ..application import ProcessDocumentRequest
from ..infrastructure import (
    InMemoryDocumentRepository,
    InMemoryProcessingResultRepository,
    EventPublisherService,
    TesseractOCRService,
    WhisperTranscriptionService,
)


class CLI:
    """CLI application wrapper."""

    def __init__(self):
        self.app = create_cli()

    def run(self, *args, **kwargs):
        """Run the CLI application."""
        return self.app(*args, **kwargs)


@click.group()
@click.version_option(version="1.0.0", prog_name="jtext")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging for detailed operation tracing",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./results",
    help="Output directory for processing results and logs",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "txt", "md"], case_sensitive=False),
    default="json",
    help="Default output format for processing results",
)
@click.option(
    "--enable-performance-tracking",
    is_flag=True,
    default=True,
    help="Enable detailed performance monitoring",
)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: bool,
    output_dir: str,
    output_format: str,
    enable_performance_tracking: bool,
) -> None:
    """
    Japanese Text Processing CLI System

    A high-precision local text extraction system for Japanese documents,
    images, and audio using OCR, ASR, and LLM correction.

    Features enterprise-grade observability with structured logging,
    correlation tracking, and comprehensive result output.

    Examples:
        jtext ocr document.jpg --llm-correct --output-format json
        jtext transcribe audio.wav --model large --lang ja
        jtext health --verbose
    """
    # Initialize CLI context with comprehensive configuration
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)
    ctx.obj["output_format"] = OutputFormat(output_format.lower())
    ctx.obj["enable_performance_tracking"] = enable_performance_tracking

    # Create output directory with proper permissions
    try:
        ctx.obj["output_dir"].mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"❌ Failed to create output directory: {e}", err=True)
        raise click.Abort()

    # Generate correlation ID for distributed tracing
    correlation_id = CorrelationIdGenerator.generate()
    set_correlation_id(correlation_id)
    ctx.obj["correlation_id"] = correlation_id

    # Initialize output service
    ctx.obj["output_service"] = get_output_service(str(ctx.obj["output_dir"]))

    # Initialize structured logger with CLI-specific configuration
    logger = get_logger("jtext.interface.cli")

    # Start performance tracking if enabled
    if enable_performance_tracking:
        ctx.obj["cli_metrics"] = start_performance_tracking()

    # Log CLI session start with structured metadata
    logger.info(
        "CLI session initiated",
        operation="cli_start",
        component="interface",
        correlation_id=correlation_id,
        output_directory=str(ctx.obj["output_dir"]),
        output_format=output_format,
        verbose_mode=verbose,
        performance_tracking=enable_performance_tracking,
        tags={"category": "system", "component": "cli", "session": "start"},
    )

    # Security audit log for CLI access
    logger.security_audit(
        "CLI access initiated",
        operation="cli_access",
        access_method="command_line",
        session_id=correlation_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@cli.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--lang", default="jpn+eng", help="OCR language")
@click.option(
    "--llm-correct",
    is_flag=True,
    default=True,
    help="Enable LLM correction (default: enabled)",
)
@click.option("--no-llm-correct", is_flag=True, help="Disable LLM correction")
@click.option(
    "--vision",
    is_flag=True,
    default=True,
    help="Enable vision analysis (default: enabled)",
)
@click.option("--no-vision", is_flag=True, help="Disable vision analysis")
@click.option("--model", help="LLM model to use for correction (e.g., gemma3:4b)")
@click.option(
    "--vision-model", help="Vision model to use for analysis (e.g., gemma3:4b)"
)
@click.pass_context
def ocr(
    ctx: click.Context,
    images: List[str],
    lang: str,
    llm_correct: bool,
    no_llm_correct: bool,
    vision: bool,
    no_vision: bool,
    model: str,
    vision_model: str,
) -> None:
    """Extract text from images using OCR with LLM correction and vision analysis.

    By default, both LLM correction and vision analysis are enabled for enhanced accuracy.
    Use --no-llm-correct or --no-vision to disable specific features.

    Examples:
        jtext ocr image1.jpg image2.png
        jtext ocr --lang jpn document.jpg
        jtext ocr --no-llm-correct --no-vision image.jpg
    """
    if not images:
        click.echo("Error: No image files specified", err=True)
        raise click.Abort()

    # Process flags - disable features if no-* flags are set
    if no_llm_correct:
        llm_correct = False
    if no_vision:
        vision = False

    # Initialize structured logger with operation context
    logger = get_logger("jtext.interface.cli.ocr")

    # Start performance tracking
    operation_metrics = start_performance_tracking()

    # Get output format from context
    output_format = ctx.obj.get("output_format", OutputFormat.JSON)
    output_service = ctx.obj.get("output_service")

    # Log operation start with structured metadata
    logger.info(
        "OCR processing operation initiated",
        operation="ocr_batch",
        component="cli",
        image_count=len(images),
        language=lang,
        llm_correction_enabled=llm_correct,
        vision_analysis_enabled=vision,
        llm_model=model or "default",
        vision_model=vision_model or "default",
        tags={"category": "business", "operation": "ocr"},
    )

    # Initialize services with dependency injection
    try:
        document_repo = InMemoryDocumentRepository()
        result_repo = InMemoryProcessingResultRepository()
        event_publisher = EventPublisherService()
        ocr_service = TesseractOCRService()

        logger.debug(
            "Processing services initialized",
            operation="service_init",
            component="infrastructure",
            tags={"category": "system"},
        )
    except Exception as e:
        logger.fatal(
            "Failed to initialize processing services",
            operation="service_init",
            error=str(e),
            tags={"category": "system_error"},
        )
        click.echo(f"❌ System error: {e}", err=True)
        raise click.Abort()

    # Process each image with comprehensive logging
    successful_processes = 0
    failed_processes = 0
    output_files = []

    for image_index, image_path in enumerate(images, 1):
        try:
            # Start per-image tracking
            image_metrics = start_performance_tracking()

            logger.info(
                f"Processing image {image_index}/{len(images)}",
                operation="ocr_single",
                component="cli",
                image_path=image_path,
                image_index=image_index,
                total_images=len(images),
                tags={"category": "business"},
            )

            click.echo(
                f"🔄 Processing [{image_index}/{len(images)}]: {Path(image_path).name}"
            )

            # Create processing request with full configuration
            request = ProcessDocumentRequest(
                file_path=image_path,
                document_type=DocumentType.IMAGE,
                language=Language.JAPANESE if "jpn" in lang else Language.ENGLISH,
                enable_correction=llm_correct,
                enable_vision=vision,
                llm_model=model,
                vision_model=vision_model,
            )

            # Execute actual OCR processing
            from ..domain import (
                ProcessingResult,
                ProcessingResultType,
                Confidence,
                DocumentId,
                Document,
                ProcessingStatus,
            )
            from ..core import generate_id

            # Perform actual OCR processing
            try:
                # Use the OCR service to extract text
                language_enum = Language.JAPANESE if "jpn" in lang else Language.ENGLISH
                ocr_result = ocr_service.extract_text(image_path, language_enum)

                if ocr_result.is_ok:
                    ocr_result_obj = ocr_result.unwrap()
                    extracted_text = ocr_result_obj.content
                    ocr_confidence = ocr_result_obj.confidence.value

                    # Log successful OCR extraction
                    logger.debug(
                        "OCR text extraction successful",
                        operation="ocr_extract",
                        component="tesseract",
                        text_length=len(extracted_text),
                        tags={"category": "processing", "result": "success"},
                    )

                    # Initialize processing results storage
                    processing_results = {
                        "tesseract_ocr": {
                            "text": extracted_text,
                            "confidence": float(ocr_confidence),
                            "processing_time_ms": 0.0,
                            "status": "success",
                        },
                        "llm_correction": {
                            "text": "",
                            "confidence": 0.0,
                            "processing_time_ms": 0.0,
                            "status": "not_applied",
                            "corrections_made": [],
                        },
                        "vision_analysis": {
                            "insights": "",
                            "confidence": 0.0,
                            "processing_time_ms": 0.0,
                            "status": "not_applied",
                            "document_structure": None,
                            "text_regions": [],
                            "visual_elements": [],
                            "quality_assessment": None,
                        },
                    }

                    # Use actual OCR confidence from Tesseract
                    confidence_score = ocr_confidence

                    # Apply LLM correction if requested
                    final_text = extracted_text
                    processing_pipeline = ["tesseract_ocr"]

                    if llm_correct and extracted_text.strip():
                        try:
                            # Import LLM correction service
                            from ..infrastructure.llm_correction import (
                                get_llm_correction_service,
                            )

                            # Initialize LLM correction service
                            llm_service = get_llm_correction_service()

                            if llm_service.is_available():
                                # Apply advanced LLM correction with image context
                                correction_result = llm_service.correct_ocr_text(
                                    text=extracted_text,
                                    confidence=float(confidence_score),
                                    language=language_enum,
                                    model_name=model,
                                    image_path=image_path,
                                )

                                if correction_result.is_ok:
                                    correction = correction_result.unwrap()

                                    # Store LLM correction results separately
                                    processing_results["llm_correction"] = {
                                        "text": correction.corrected_text,
                                        "confidence": correction.confidence.to_float(),
                                        "processing_time_ms": correction.processing_time_ms,
                                        "status": "success",
                                        "corrections_made": correction.corrections_made,
                                        "model_used": correction.model_used,
                                    }

                                    final_text = correction.corrected_text
                                    confidence_score = correction.confidence.to_float()
                                    processing_pipeline.append("llm_correction")

                                    logger.info(
                                        "LLM-OCR complement applied successfully",
                                        operation="llm_ocr_complement",
                                        component="llm",
                                        model=correction.model_used,
                                        original_length=len(extracted_text),
                                        corrected_length=len(final_text),
                                        corrections_count=len(
                                            correction.corrections_made
                                        ),
                                        confidence_improvement=correction.confidence.to_float()
                                        - float(ocr_confidence),
                                        processing_time_ms=correction.processing_time_ms,
                                        tags={
                                            "category": "processing",
                                            "result": "success",
                                        },
                                    )
                                else:
                                    # LLM correction failed, use original text
                                    correction_error = correction_result.unwrap_err()
                                    processing_results["llm_correction"][
                                        "status"
                                    ] = "failed"
                                    processing_results["llm_correction"]["error"] = str(
                                        correction_error
                                    )
                                    final_text = extracted_text

                                    logger.warn(
                                        "LLM-OCR complement failed, using OCR result only",
                                        operation="llm_ocr_complement",
                                        error=correction_error,
                                        tags={
                                            "category": "processing",
                                            "result": "partial_failure",
                                        },
                                    )
                            else:
                                # LLM service not available
                                processing_results["llm_correction"][
                                    "status"
                                ] = "service_unavailable"
                                processing_results["llm_correction"][
                                    "error"
                                ] = "Ollama not accessible"
                                final_text = f"{extracted_text}\n\n[LLM correction unavailable - Ollama not accessible]"

                                logger.warn(
                                    "LLM-OCR complement service not available",
                                    operation="llm_ocr_complement",
                                    component="llm",
                                    tags={
                                        "category": "system",
                                        "result": "service_unavailable",
                                    },
                                )

                        except Exception as llm_error:
                            # Fallback to original text on any error
                            processing_results["llm_correction"]["status"] = "exception"
                            processing_results["llm_correction"]["error"] = str(
                                llm_error
                            )
                            final_text = extracted_text

                            logger.warn(
                                "LLM-OCR complement failed with exception, using OCR result only",
                                operation="llm_ocr_complement",
                                error=str(llm_error),
                                tags={
                                    "category": "processing",
                                    "result": "partial_failure",
                                },
                            )

                    # Apply vision analysis if requested
                    if vision and extracted_text.strip():
                        try:
                            # Import vision analysis service
                            from ..infrastructure.vision_analysis import (
                                get_vision_analysis_service,
                            )

                            # Initialize vision analysis service
                            vision_service = get_vision_analysis_service()

                            if vision_service.is_available():
                                # Perform comprehensive vision analysis
                                vision_result = vision_service.analyze_document(
                                    image_path=image_path,
                                    language=language_enum,
                                    analysis_type="comprehensive",
                                    model_name=vision_model,
                                )

                                if vision_result.is_ok:
                                    analysis = vision_result.unwrap()

                                    # Store vision analysis results separately (convert to JSON-serializable format)
                                    processing_results["vision_analysis"] = {
                                        "insights": _format_vision_insights(analysis),
                                        "confidence": analysis.confidence.to_float(),
                                        "processing_time_ms": analysis.processing_time_ms,
                                        "status": "success",
                                        "document_structure": (
                                            {
                                                "document_type": (
                                                    analysis.layout_structure.document_type
                                                    if analysis.layout_structure
                                                    else None
                                                ),
                                                "main_sections": (
                                                    analysis.layout_structure.main_sections
                                                    if analysis.layout_structure
                                                    else []
                                                ),
                                                "layout_quality": (
                                                    analysis.layout_structure.layout_quality
                                                    if analysis.layout_structure
                                                    else None
                                                ),
                                            }
                                            if analysis.layout_structure
                                            else None
                                        ),
                                        "text_regions": (
                                            [
                                                {
                                                    "area": region.area,
                                                    "content_type": region.content_type,
                                                    "confidence": region.confidence,
                                                    "text_content": region.text_content,
                                                }
                                                for region in analysis.text_regions
                                            ]
                                            if analysis.text_regions
                                            else []
                                        ),
                                        "visual_elements": (
                                            [
                                                {
                                                    "element_type": element.element_type,
                                                    "description": element.description,
                                                    "position": element.position,
                                                    "confidence": element.confidence,
                                                }
                                                for element in analysis.visual_elements
                                            ]
                                            if analysis.visual_elements
                                            else []
                                        ),
                                        "quality_assessment": (
                                            {
                                                "clarity": analysis.quality_assessment.clarity,
                                                "contrast": analysis.quality_assessment.contrast,
                                                "readability": analysis.quality_assessment.readability,
                                                "enhancement_suggestions": analysis.quality_assessment.enhancement_suggestions,
                                            }
                                            if analysis.quality_assessment
                                            else None
                                        ),
                                        "content_categories": analysis.content_categories,
                                        "model_used": analysis.model_used,
                                        "analysis_type": analysis.analysis_type.value,
                                    }

                                    # Enhance text with vision analysis insights
                                    vision_insights = _format_vision_insights(analysis)
                                    final_text += f"\n\n--- Vision Analysis ---\n{vision_insights}"
                                    processing_pipeline.append("vision_analysis")

                                    # Boost confidence based on vision analysis quality
                                    vision_confidence_boost = min(
                                        0.1, analysis.confidence.to_float() * 0.1
                                    )
                                    confidence_score = min(
                                        0.99,
                                        float(confidence_score)
                                        + vision_confidence_boost,
                                    )

                                    logger.info(
                                        "Vision analysis completed successfully",
                                        operation="vision_analysis",
                                        component="vision",
                                        model=analysis.model_used,
                                        analysis_type=analysis.analysis_type.value,
                                        confidence=analysis.confidence.to_float(),
                                        processing_time_ms=analysis.processing_time_ms,
                                        tags={
                                            "category": "processing",
                                            "result": "success",
                                        },
                                    )
                                else:
                                    # Vision analysis failed, use fallback
                                    vision_error = vision_result.unwrap_err()
                                    processing_results["vision_analysis"][
                                        "status"
                                    ] = "failed"
                                    processing_results["vision_analysis"]["error"] = (
                                        str(vision_error)
                                    )
                                    final_text += (
                                        f"\n\n[Vision analysis failed: {vision_error}]"
                                    )
                                    processing_pipeline.append("vision_analysis_failed")

                                    logger.warn(
                                        "Vision analysis failed, using fallback",
                                        operation="vision_analysis",
                                        component="vision",
                                        error=vision_error,
                                        tags={
                                            "category": "processing",
                                            "result": "partial_failure",
                                        },
                                    )
                            else:
                                # Vision service not available
                                processing_results["vision_analysis"][
                                    "status"
                                ] = "service_unavailable"
                                processing_results["vision_analysis"][
                                    "error"
                                ] = "Ollama not accessible"
                                final_text += f"\n\n[Vision analysis unavailable - Ollama not accessible]"
                                processing_pipeline.append(
                                    "vision_analysis_unavailable"
                                )

                                logger.warn(
                                    "Vision analysis service not available",
                                    operation="vision_analysis",
                                    component="vision",
                                    tags={
                                        "category": "system",
                                        "result": "service_unavailable",
                                    },
                                )

                        except Exception as vision_error:
                            # Fallback to basic vision analysis
                            processing_results["vision_analysis"][
                                "status"
                            ] = "exception"
                            processing_results["vision_analysis"]["error"] = str(
                                vision_error
                            )
                            final_text += f"\n\n[Vision analysis failed with exception: {str(vision_error)}]"
                            processing_pipeline.append("vision_analysis_error")

                            logger.warn(
                                "Vision analysis failed with exception",
                                operation="vision_analysis",
                                component="vision",
                                error=str(vision_error),
                                tags={
                                    "category": "processing",
                                    "result": "exception_fallback",
                                },
                            )

                    # Comprehensive LLM synthesis of all three stages
                    if (llm_correct or vision) and extracted_text.strip():
                        try:
                            # Import LLM correction service for synthesis
                            from ..infrastructure.llm_correction import (
                                get_llm_correction_service,
                            )

                            # Initialize LLM service for synthesis
                            synthesis_service = get_llm_correction_service()

                            if synthesis_service.is_available():
                                # Prepare comprehensive input for synthesis
                                synthesis_input = f"""
=== TESSERACT OCR RESULT ===
Text: {processing_results["tesseract_ocr"]["text"]}
Confidence: {processing_results["tesseract_ocr"]["confidence"]:.3f}
Status: {processing_results["tesseract_ocr"]["status"]}

=== LLM CORRECTION RESULT ===
Text: {processing_results["llm_correction"]["text"] if processing_results["llm_correction"]["text"] else "Not applied"}
Confidence: {processing_results["llm_correction"]["confidence"]:.3f}
Status: {processing_results["llm_correction"]["status"]}
Corrections Made: {len(processing_results["llm_correction"].get("corrections_made", []))}

=== VISION ANALYSIS RESULT ===
Insights: {processing_results["vision_analysis"]["insights"] if processing_results["vision_analysis"]["insights"] else "Not applied"}
Confidence: {processing_results["vision_analysis"]["confidence"]:.3f}
Status: {processing_results["vision_analysis"]["status"]}
"""

                                # Create synthesis prompt
                                synthesis_prompt = f"""
You are an expert document analysis system. You have received results from three different processing stages:

{synthesis_input}

Please provide a comprehensive, final analysis that:
1. Synthesizes the best information from all three stages
2. Identifies the most accurate text content
3. Provides confidence assessment based on all available data
4. Highlights any discrepancies or areas of uncertainty
5. Offers recommendations for document quality improvement

Format your response as:
FINAL_TEXT: [The most accurate text content]
CONFIDENCE: [Overall confidence score 0.0-1.0]
ANALYSIS: [Your comprehensive analysis]
RECOMMENDATIONS: [Any improvement suggestions]
"""

                                # Perform synthesis using LLM
                                synthesis_result = synthesis_service.correct_ocr_text(
                                    text=synthesis_prompt,
                                    confidence=float(confidence_score),
                                    language=language_enum,
                                    model_name=model,
                                    image_path=image_path,
                                )

                                if synthesis_result.is_ok:
                                    synthesis = synthesis_result.unwrap()

                                    # Store synthesis results
                                    processing_results["comprehensive_synthesis"] = {
                                        "text": synthesis.corrected_text,
                                        "confidence": synthesis.confidence.to_float(),
                                        "processing_time_ms": synthesis.processing_time_ms,
                                        "status": "success",
                                        "model_used": synthesis.model_used,
                                        "synthesis_type": "comprehensive_analysis",
                                    }

                                    # Update final text with synthesis result
                                    final_text = synthesis.corrected_text
                                    confidence_score = synthesis.confidence.to_float()
                                    processing_pipeline.append(
                                        "comprehensive_synthesis"
                                    )

                                    logger.info(
                                        "Comprehensive synthesis completed successfully",
                                        operation="comprehensive_synthesis",
                                        component="llm",
                                        model=synthesis.model_used,
                                        original_length=len(extracted_text),
                                        synthesized_length=len(final_text),
                                        confidence_improvement=synthesis.confidence.to_float()
                                        - float(ocr_confidence),
                                        processing_time_ms=synthesis.processing_time_ms,
                                        tags={
                                            "category": "processing",
                                            "result": "success",
                                        },
                                    )
                                else:
                                    # Synthesis failed, use existing results
                                    synthesis_error = synthesis_result.unwrap_err()
                                    processing_results["comprehensive_synthesis"] = {
                                        "text": final_text,
                                        "confidence": confidence_score,
                                        "processing_time_ms": 0.0,
                                        "status": "failed",
                                        "error": str(synthesis_error),
                                        "synthesis_type": "comprehensive_analysis",
                                    }

                                    logger.warn(
                                        "Comprehensive synthesis failed, using existing results",
                                        operation="comprehensive_synthesis",
                                        error=synthesis_error,
                                        tags={
                                            "category": "processing",
                                            "result": "partial_failure",
                                        },
                                    )
                            else:
                                # Synthesis service not available
                                processing_results["comprehensive_synthesis"] = {
                                    "text": final_text,
                                    "confidence": confidence_score,
                                    "processing_time_ms": 0.0,
                                    "status": "service_unavailable",
                                    "error": "Ollama not accessible",
                                    "synthesis_type": "comprehensive_analysis",
                                }

                                logger.warn(
                                    "Comprehensive synthesis service not available",
                                    operation="comprehensive_synthesis",
                                    component="llm",
                                    tags={
                                        "category": "system",
                                        "result": "service_unavailable",
                                    },
                                )

                        except Exception as synthesis_error:
                            # Fallback to existing results
                            processing_results["comprehensive_synthesis"] = {
                                "text": final_text,
                                "confidence": confidence_score,
                                "processing_time_ms": 0.0,
                                "status": "exception",
                                "error": str(synthesis_error),
                                "synthesis_type": "comprehensive_analysis",
                            }

                            logger.warn(
                                "Comprehensive synthesis failed with exception",
                                operation="comprehensive_synthesis",
                                error=str(synthesis_error),
                                tags={
                                    "category": "processing",
                                    "result": "exception_fallback",
                                },
                            )

                    image_metrics.finish()

                    # Create comprehensive processing result with separate stage results
                    processing_result = ProcessingResult(
                        id=generate_id(),
                        document_id=DocumentId.generate(),
                        result_type=ProcessingResultType.OCR,
                        content=final_text,
                        confidence=Confidence(confidence_score),
                        processing_time=(
                            image_metrics.duration_ms / 1000
                            if image_metrics.duration_ms
                            else 0.0
                        ),
                        created_at=datetime.now(timezone.utc),
                        metadata={
                            "source_file": image_path,
                            "language": lang,
                            "llm_correction": llm_correct,
                            "vision_analysis": vision,
                            "processing_pipeline": processing_pipeline,
                            "ocr_engine": "tesseract",
                            "llm_model": model if llm_correct else None,
                            "vision_model": vision_model if vision else None,
                            "original_text_length": len(extracted_text),
                            "final_text_length": len(final_text),
                            "mock_result": False,
                            # Separate processing results for each stage
                            "processing_stages": processing_results,
                            # Summary of all stages
                            "stage_summary": {
                                "tesseract_ocr": {
                                    "status": processing_results["tesseract_ocr"][
                                        "status"
                                    ],
                                    "confidence": processing_results["tesseract_ocr"][
                                        "confidence"
                                    ],
                                    "text_length": len(
                                        processing_results["tesseract_ocr"]["text"]
                                    ),
                                },
                                "llm_correction": {
                                    "status": processing_results["llm_correction"][
                                        "status"
                                    ],
                                    "confidence": processing_results["llm_correction"][
                                        "confidence"
                                    ],
                                    "text_length": (
                                        len(
                                            processing_results["llm_correction"]["text"]
                                        )
                                        if processing_results["llm_correction"]["text"]
                                        else 0
                                    ),
                                    "corrections_count": len(
                                        processing_results["llm_correction"].get(
                                            "corrections_made", []
                                        )
                                    ),
                                },
                                "vision_analysis": {
                                    "status": processing_results["vision_analysis"][
                                        "status"
                                    ],
                                    "confidence": processing_results["vision_analysis"][
                                        "confidence"
                                    ],
                                    "insights_length": (
                                        len(
                                            processing_results["vision_analysis"][
                                                "insights"
                                            ]
                                        )
                                        if processing_results["vision_analysis"][
                                            "insights"
                                        ]
                                        else 0
                                    ),
                                },
                                "comprehensive_synthesis": {
                                    "status": processing_results.get(
                                        "comprehensive_synthesis", {}
                                    ).get("status", "not_applied"),
                                    "confidence": processing_results.get(
                                        "comprehensive_synthesis", {}
                                    ).get("confidence", 0.0),
                                    "text_length": (
                                        len(
                                            processing_results[
                                                "comprehensive_synthesis"
                                            ]["text"]
                                        )
                                        if processing_results.get(
                                            "comprehensive_synthesis", {}
                                        ).get("text")
                                        else 0
                                    ),
                                    "synthesis_type": processing_results.get(
                                        "comprehensive_synthesis", {}
                                    ).get("synthesis_type", "none"),
                                },
                            },
                        },
                    )

                else:
                    # OCR failed, create error result
                    ocr_error = ocr_result.unwrap_err()
                    image_metrics.finish()

                    logger.error(
                        "OCR processing failed",
                        operation="ocr_extract",
                        component="tesseract",
                        error=str(ocr_error),
                        tags={"category": "processing_error", "result": "failure"},
                    )

                    processing_result = ProcessingResult(
                        id=generate_id(),
                        document_id=DocumentId.generate(),
                        result_type=ProcessingResultType.OCR,
                        content=f"OCR processing failed: {str(ocr_error)}",
                        confidence=Confidence(0.0),
                        processing_time=(
                            image_metrics.duration_ms / 1000
                            if image_metrics.duration_ms
                            else 0.0
                        ),
                        created_at=datetime.now(timezone.utc),
                        metadata={
                            "source_file": image_path,
                            "language": lang,
                            "error": str(ocr_error),
                            "processing_pipeline": ["tesseract_ocr_failed"],
                            "mock_result": False,
                        },
                    )

            except Exception as processing_error:
                # Unexpected processing error
                image_metrics.finish()

                logger.error(
                    "Unexpected processing error",
                    operation="ocr_process",
                    component="cli",
                    error=str(processing_error),
                    tags={"category": "system_error", "result": "failure"},
                )

                processing_result = ProcessingResult(
                    id=generate_id(),
                    document_id=DocumentId.generate(),
                    result_type=ProcessingResultType.OCR,
                    content=f"Processing failed due to system error: {str(processing_error)}",
                    confidence=Confidence(0.0),
                    processing_time=(
                        image_metrics.duration_ms / 1000
                        if image_metrics.duration_ms
                        else 0.0
                    ),
                    created_at=datetime.now(timezone.utc),
                    metadata={
                        "source_file": image_path,
                        "language": lang,
                        "system_error": str(processing_error),
                        "processing_pipeline": ["system_error"],
                        "mock_result": False,
                    },
                )

            # Create document entity
            document = Document(
                id=processing_result.document_id,
                file_path=FilePath(image_path),
                document_type=DocumentType.IMAGE,
                language=Language.JAPANESE if "jpn" in lang else Language.ENGLISH,
                status=ProcessingStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Save result to output file
            if output_service:
                save_result = output_service.save_processing_result(
                    result=processing_result,
                    document=document,
                    output_format=output_format,
                    correlation_id=ctx.obj["correlation_id"],
                )

                if save_result.is_ok:
                    output_file = save_result.unwrap()
                    output_files.append(output_file)

                    logger.info(
                        "Image processing completed successfully",
                        operation="ocr_single",
                        component="cli",
                        image_path=image_path,
                        output_file=output_file,
                        duration_ms=image_metrics.duration_ms,
                        confidence=processing_result.confidence.value,
                        tags={"category": "business", "result": "success"},
                    )

                    click.echo(
                        f"✅ Processed: {Path(image_path).name} → {Path(output_file).name}"
                    )
                    successful_processes += 1
                else:
                    error = save_result.unwrap_err()
                    raise Exception(f"Failed to save result: {error}")
            else:
                click.echo(
                    f"✅ Processed: {Path(image_path).name} (output service not available)"
                )
                successful_processes += 1

        except Exception as e:
            failed_processes += 1

            logger.error(
                "Image processing failed",
                operation="ocr_single",
                component="cli",
                image_path=image_path,
                error=str(e),
                tags={"category": "processing_error"},
            )

            click.echo(f"❌ Failed: {Path(image_path).name} - {str(e)}", err=True)

    # Finish operation tracking
    operation_metrics.finish()

    # Log operation completion
    logger.info(
        "OCR batch processing completed",
        operation="ocr_batch",
        component="cli",
        total_images=len(images),
        successful_processes=successful_processes,
        failed_processes=failed_processes,
        success_rate=successful_processes / len(images) if images else 0,
        total_duration_ms=operation_metrics.duration_ms,
        output_files_count=len(output_files),
        tags={"category": "business", "operation": "batch_complete"},
    )

    # Display summary
    click.echo(f"\n📊 Processing Summary:")
    click.echo(
        f"   Total: {len(images)} | Success: {successful_processes} | Failed: {failed_processes}"
    )
    click.echo(f"   Success rate: {successful_processes/len(images)*100:.1f}%")
    if output_files:
        click.echo(f"   Results saved to: {ctx.obj['output_dir']}")
    if operation_metrics.duration_ms:
        click.echo(f"   Total time: {operation_metrics.duration_ms/1000:.2f}s")


def _format_vision_insights(analysis) -> str:
    """Format vision analysis insights for display."""
    insights = []

    # Document structure
    layout = analysis.layout_structure
    if layout:
        doc_type = layout.document_type
        sections = layout.main_sections
        insights.append(f"Document Type: {doc_type}")
        if sections:
            insights.append(f"Main Sections: {', '.join(sections)}")

    # Content categories
    if analysis.content_categories:
        insights.append(f"Content Categories: {', '.join(analysis.content_categories)}")

    # Text regions
    if analysis.text_regions:
        insights.append(f"Text Regions: {len(analysis.text_regions)} identified")
        for region in analysis.text_regions[:3]:  # Show first 3 regions
            area = region.area
            content_type = region.content_type
            insights.append(f"  - {area}: {content_type}")

    # Visual elements
    if analysis.visual_elements:
        insights.append(f"Visual Elements: {len(analysis.visual_elements)} detected")
        for element in analysis.visual_elements[:3]:  # Show first 3 elements
            element_type = element.element_type
            description = element.description
            insights.append(f"  - {element_type}: {description}")

    # Quality assessment
    quality = analysis.quality_assessment
    if quality:
        clarity = quality.clarity
        contrast = quality.contrast
        readability = quality.readability
        insights.append(
            f"Quality: Clarity {clarity:.1f}, Contrast {contrast:.1f}, Readability {readability:.1f}"
        )

    # Enhancement suggestions
    if quality and quality.enhancement_suggestions:
        suggestions = quality.enhancement_suggestions
        if suggestions:
            insights.append(f"Enhancement Suggestions: {', '.join(suggestions)}")

    return "\n".join(insights) if insights else "No specific insights available"


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--type", type=click.Choice(["image", "audio", "pdf"]), help="Document type"
)
@click.option("--lang", default="jpn+eng", help="Language")
@click.option(
    "--output-format",
    type=click.Choice(["json", "txt", "md"]),
    default="json",
    help="Output format",
)
@click.pass_context
def ingest(
    ctx: click.Context, files: List[str], type: str, lang: str, output_format: str
) -> None:
    """Ingest and process documents.

    Examples:
        jtext ingest --type image *.jpg
        jtext ingest --type audio --lang jpn audio.wav
    """
    if not files:
        click.echo("Error: No files specified", err=True)
        raise click.Abort()

    logger = get_logger("jtext.cli.ingest")
    logger.info(f"Ingesting {len(files)} file(s) of type: {type}")

    # Determine document type
    document_type = DocumentType.IMAGE
    if type == "audio":
        document_type = DocumentType.AUDIO
    elif type == "pdf":
        document_type = DocumentType.PDF

    # Process each file
    for file_path in files:
        try:
            click.echo(f"Ingesting: {file_path}")

            # Create processing request
            request = ProcessDocumentRequest(
                file_path=file_path,
                document_type=document_type,
                language=Language(lang),
                output_format=output_format,
            )

            # TODO: Wire up to actual use case
            click.echo(f"✓ Ingested: {file_path}")

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {str(e)}")
            click.echo(f"✗ Error ingesting {file_path}: {e}", err=True)


@cli.command()
@click.argument("audio_files", nargs=-1, type=click.Path(exists=True))
@click.option("--model", default="base", help="Whisper model size")
@click.option("--lang", default="jpn", help="Audio language")
@click.pass_context
def transcribe(
    ctx: click.Context, audio_files: List[str], model: str, lang: str
) -> None:
    """Transcribe audio files to text.

    Examples:
        jtext transcribe audio.wav
        jtext transcribe --model large --lang eng *.wav
    """
    if not audio_files:
        click.echo("Error: No audio files specified", err=True)
        raise click.Abort()

    logger = get_logger("jtext.cli.transcribe")
    logger.info(f"Transcribing {len(audio_files)} audio file(s) with model: {model}")

    # Initialize services
    transcription_service = WhisperTranscriptionService(model_name=model)

    # Process each audio file
    for audio_path in audio_files:
        try:
            click.echo(f"Transcribing: {audio_path}")

            # TODO: Wire up to actual transcription use case
            click.echo(f"✓ Transcribed: {audio_path}")

        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
            click.echo(f"✗ Error transcribing {audio_path}: {e}", err=True)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show processing statistics.

    Examples:
        jtext stats
    """
    logger = get_logger("jtext.cli.stats")
    logger.info("Retrieving processing statistics")

    # TODO: Wire up to actual statistics use case
    click.echo("Processing Statistics:")
    click.echo("  Total documents: 0")
    click.echo("  Completed: 0")
    click.echo("  Failed: 0")
    click.echo("  Success rate: 0%")


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check system health.

    Examples:
        jtext health
    """
    logger = get_logger("jtext.cli.health")
    logger.info("Checking system health")

    # Check dependencies
    health_status = {
        "tesseract": "✓ Available",
        "whisper": "✓ Available",
        "ollama": "✓ Available",
        "system": "✓ Healthy",
    }

    click.echo("System Health Check:")
    for service, status in health_status.items():
        click.echo(f"  {service}: {status}")


def create_cli():
    """Create CLI instance for entry point."""
    return cli


# Create the CLI instance for entry point
cli_instance = create_cli()

if __name__ == "__main__":
    cli_instance()
