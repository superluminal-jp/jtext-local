"""
Main CLI entry point for jtext system.

This module provides the Click-based command-line interface for the
Japanese text processing system.
"""

import click
from pathlib import Path
from typing import List, Optional

from .core.ocr_hybrid import HybridOCR
from .core.multimodal_ocr import MultimodalOCR
from .processing.document_extractor import DocumentExtractor
from .transcription.audio_transcriber import AudioTranscriber
from .utils.logging import setup_logging
from .utils.io_utils import ensure_output_dir, save_results
from .utils.validation import (
    validate_image_file,
    validate_document_file,
    validate_audio_file,
)
from .config.settings import Settings


@click.group()
@click.version_option(version="1.0.0", prog_name="jtext")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./out",
    help="Output directory for results",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, output_dir: str) -> None:
    """
    Japanese Text Processing CLI System

    High-precision local text extraction for Japanese documents,
    images, and audio using OCR, ASR, and LLM correction.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(verbose=verbose)

    # Store global options in context
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)

    # Ensure output directory exists
    ensure_output_dir(ctx.obj["output_dir"])


@cli.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--lang", default="jpn+eng", help="OCR language (default: jpn+eng)")
@click.option("--llm-correct", is_flag=True, help="Enable LLM correction")
@click.option("--vision", is_flag=True, help="Enable vision analysis")
@click.option("--model", default="gpt-oss", help="LLM model for correction")
@click.option("--vision-model", default="llava", help="Vision model for image analysis")
@click.pass_context
def ocr(
    ctx: click.Context,
    images: List[str],
    lang: str,
    llm_correct: bool,
    vision: bool,
    model: str,
    vision_model: str,
) -> None:
    """
    Extract text from images using advanced OCR with vision analysis and LLM correction.

    IMAGES: One or more image files to process
    """
    if not images:
        click.echo("Error: No image files specified", err=True)
        raise click.Abort()

    click.echo(
        f"Processing {len(images)} image(s) with {'multimodal ' if vision else ''}OCR..."
    )

    # Initialize OCR processor
    if vision:
        ocr_processor = MultimodalOCR(
            llm_model=model if llm_correct else None,
            vision_model=vision_model,
            enable_correction=llm_correct,
            enable_vision=vision,
        )
        click.echo(f"Using multimodal OCR with vision model: {vision_model}")
    else:
        ocr_processor = HybridOCR(
            llm_model=model if llm_correct else None, enable_correction=llm_correct
        )

    # Process each image
    for image_path in images:
        try:
            click.echo(f"Processing: {image_path}")
            result = ocr_processor.process_image(image_path)

            # Save results
            output_path = ctx.obj["output_dir"] / Path(image_path).stem
            save_results(result, output_path)

            # Show processing details
            if hasattr(result, "fusion_method"):
                click.echo(f"  Fusion method: {result.fusion_method}")
            if hasattr(result, "vision_analysis") and result.vision_analysis:
                click.echo(
                    f"  Vision analysis: {result.vision_analysis.get('model', 'N/A')}"
                )

            click.echo(f"✓ Completed: {output_path}.txt")

        except Exception as e:
            click.echo(f"✗ Error processing {image_path}: {e}", err=True)
            if ctx.obj["verbose"]:
                raise
            else:
                raise click.Abort()


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--fallback-ocr", is_flag=True, help="Use OCR fallback for low-quality pages"
)
@click.option("--llm-correct", is_flag=True, help="Enable LLM correction")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./out",
    help="Output directory for results",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    files: List[str],
    fallback_ocr: bool,
    llm_correct: bool,
    output_dir: str,
) -> None:
    """
    Extract text from structured documents with optional OCR fallback.

    FILES: One or more document files to process (PDF, DOCX, PPTX, HTML)
    """
    if not files:
        click.echo("Error: No files specified", err=True)
        raise click.Abort()

    click.echo(f"Processing {len(files)} document(s)...")

    # Initialize document extractor
    extractor = DocumentExtractor()

    # Process each document
    for file_path in files:
        try:
            click.echo(f"Processing: {file_path}")
            result = extractor.extract_text(file_path)

            # Save results
            output_path = ctx.obj["output_dir"] / Path(file_path).stem
            save_results(result, output_path)

            click.echo(f"✓ Completed: {output_path}.txt")

        except Exception as e:
            click.echo(f"✗ Error processing {file_path}: {e}", err=True)
            if ctx.obj["verbose"]:
                raise
            else:
                raise click.Abort()


@cli.command()
@click.argument("audio_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--model",
    default="base",
    help="Whisper model size (tiny, base, small, medium, large)",
)
@click.option("--lang", default="ja", help="Language code (default: ja)")
@click.option("--llm-correct", is_flag=True, help="Enable LLM correction")
@click.pass_context
def transcribe(
    ctx: click.Context, audio_files: List[str], model: str, lang: str, llm_correct: bool
) -> None:
    """
    Transcribe audio/video files to text using ASR.

    AUDIO_FILES: One or more audio/video files to process
    """
    if not audio_files:
        click.echo("Error: No audio files specified", err=True)
        raise click.Abort()

    click.echo(f"Processing {len(audio_files)} audio file(s)...")

    # Initialize audio transcriber
    transcriber = AudioTranscriber(model_size=model)

    # Process each audio file
    for audio_path in audio_files:
        try:
            click.echo(f"Processing: {audio_path}")
            result = transcriber.transcribe_audio(audio_path, language=lang)

            # Save results
            output_path = ctx.obj["output_dir"] / Path(audio_path).stem
            save_results(result, output_path)

            click.echo(f"✓ Completed: {output_path}.txt")

        except Exception as e:
            click.echo(f"✗ Error processing {audio_path}: {e}", err=True)
            if ctx.obj["verbose"]:
                raise
            else:
                raise click.Abort()


@cli.command()
@click.option("--prompt", "-p", required=True, help="Text prompt for LLM")
@click.option("--model", default="gpt-oss", help="LLM model to use")
@click.option(
    "--context", "-c", type=click.Path(exists=True), help="Context file to include"
)
@click.pass_context
def chat(ctx: click.Context, prompt: str, model: str, context: Optional[str]) -> None:
    """
    Interact with LLM for text processing tasks.
    """
    click.echo("LLM chat functionality not yet implemented in MVP")
    click.echo("Use 'ocr' command for image processing")


if __name__ == "__main__":
    cli()
