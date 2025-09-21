#!/usr/bin/env python3
"""
Phase 2 Demo Script for jtext System

This script demonstrates the new Phase 2 features including:
- Enhanced LLM integration with Ollama
- Document processing (PDF, DOCX, PPTX, HTML)
- Audio transcription with Whisper
- Context-aware correction
"""

import os
import sys
import time
from pathlib import Path
import tempfile
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from jtext.core.ocr_hybrid import HybridOCR, ProcessingResult
from jtext.processing.document_extractor import DocumentExtractor
from jtext.transcription.audio_transcriber import AudioTranscriber
from jtext.correction.context_aware_corrector import ContextAwareCorrector
from jtext.utils.logging import setup_logging
from jtext.utils.io_utils import ensure_output_dir, save_results


def create_sample_documents():
    """Create sample documents for testing."""
    print("📄 Creating sample documents...")

    # Create a sample text file
    sample_text = """
    システム要求定義書
    
    1. プロジェクト概要
    このプロジェクトは、日本語テキスト処理システムの開発を目的としています。
    
    2. 機能要件
    - OCR機能: 画像からテキストを抽出
    - ASR機能: 音声からテキストを変換
    - LLM修正: 抽出されたテキストの精度向上
    
    3. 非機能要件
    - 処理速度: 1ページあたり5秒以内
    - 精度: 95%以上
    - 対応言語: 日本語、英語
    """

    # Create sample HTML
    sample_html = """
    <html>
    <head><title>日本語テキスト処理システム</title></head>
    <body>
        <h1>システム概要</h1>
        <p>このシステムは、日本語テキストの高精度処理を提供します。</p>
        <ul>
            <li>OCR機能</li>
            <li>ASR機能</li>
            <li>LLM修正</li>
        </ul>
        <img src="sample.jpg" alt="サンプル画像">
        <a href="https://example.com">詳細情報</a>
    </body>
    </html>
    """

    # Create temporary files
    temp_dir = Path(tempfile.mkdtemp())

    # Text file
    text_file = temp_dir / "sample.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    # HTML file
    html_file = temp_dir / "sample.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(sample_html)

    print(f"✅ Created sample documents in: {temp_dir}")
    return temp_dir, [text_file, html_file]


def demo_enhanced_ocr():
    """Demonstrate enhanced OCR with LLM correction."""
    print("\n🔍 Enhanced OCR with LLM Correction Demo")
    print("=" * 50)

    # Initialize OCR processor with LLM correction
    ocr_processor = HybridOCR(enable_correction=True, llm_model="llama2")

    # Check if we have a sample image
    sample_image = Path("sample_image.png")
    if not sample_image.exists():
        print("⚠️  No sample image found. Please add a sample_image.png file.")
        return

    try:
        print(f"Processing image: {sample_image}")
        result = ocr_processor.process_image(str(sample_image))

        print(f"✅ OCR completed in {result.processing_time:.2f}s")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"🔧 Corrections applied: {result.corrections_applied}")
        print(f"📝 Text length: {len(result.text)} characters")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        save_results(result, output_dir / "enhanced_ocr")

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ OCR processing failed: {e}")


def demo_document_processing():
    """Demonstrate document processing capabilities."""
    print("\n📄 Document Processing Demo")
    print("=" * 50)

    # Create sample documents
    temp_dir, sample_files = create_sample_documents()

    # Initialize document extractor
    extractor = DocumentExtractor()

    for file_path in sample_files:
        try:
            print(f"Processing: {file_path.name}")
            result = extractor.extract_text(str(file_path))

            print(f"✅ Extraction completed in {result.processing_time:.2f}s")
            print(f"📝 Text length: {len(result.text)} characters")

            # Show document metadata
            if hasattr(result, "document_metadata"):
                metadata = result.document_metadata
                print(f"📊 Document format: {metadata.get('format', 'unknown')}")
                if "pages" in metadata:
                    print(f"📄 Pages: {metadata['pages']}")
                if "paragraphs" in metadata:
                    print(f"📝 Paragraphs: {metadata['paragraphs']}")

            # Save results
            output_dir = Path("demo_output")
            ensure_output_dir(output_dir)
            save_results(result, output_dir / file_path.stem)

        except Exception as e:
            print(f"❌ Document processing failed: {e}")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


def demo_audio_transcription():
    """Demonstrate audio transcription capabilities."""
    print("\n🎤 Audio Transcription Demo")
    print("=" * 50)

    # Initialize audio transcriber
    transcriber = AudioTranscriber(model_size="base")

    # Check for sample audio files
    audio_extensions = [".mp3", ".wav", ".m4a", ".mp4"]
    sample_audio = None

    for ext in audio_extensions:
        audio_file = Path(f"sample_audio{ext}")
        if audio_file.exists():
            sample_audio = audio_file
            break

    if not sample_audio:
        print("⚠️  No sample audio files found. Please add a sample audio file.")
        print("Supported formats: .mp3, .wav, .m4a, .mp4")
        return

    try:
        print(f"Processing audio: {sample_audio}")
        result = transcriber.transcribe_audio(str(sample_audio), language="ja")

        print(f"✅ Transcription completed in {result.processing_time:.2f}s")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"📝 Text length: {len(result.text)} characters")

        # Show audio metadata
        if hasattr(result, "audio_metadata"):
            metadata = result.audio_metadata
            print(f"🌍 Language: {metadata.get('language', 'unknown')}")
            print(f"⏱️  Duration: {metadata.get('duration', 0):.2f}s")
            print(f"🤖 Model: {metadata.get('model_size', 'unknown')}")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        save_results(result, output_dir / "audio_transcription")

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Audio transcription failed: {e}")


def demo_context_aware_correction():
    """Demonstrate context-aware correction."""
    print("\n🧠 Context-Aware Correction Demo")
    print("=" * 50)

    # Initialize context-aware corrector
    corrector = ContextAwareCorrector(model="llama2")

    # Sample OCR text with errors
    sample_ocr_text = """
    システム要求定義書
    
    1. プロジェクト概要
    このプロジェクトは、日本語テキスト処理システムの開発を目的としています。
    
    2. 機能要件
    - OCR機能: 画像からテキストを抽出
    - ASR機能: 音声からテキストを変換
    - LLM修正: 抽出されたテキストの精度向上
    
    3. 非機能要件
    - 処理速度: 1ページあたり5秒以内
    - 精度: 95%以上
    - 対応言語: 日本語、英語
    """

    # Document metadata for context
    document_metadata = {
        "format": "pdf",
        "pages": 3,
        "has_tables": True,
        "has_images": False,
        "language": "ja",
    }

    try:
        print("Applying context-aware correction...")
        corrected_text, corrections = corrector.correct_with_context(
            text=sample_ocr_text,
            context_type="technical",
            document_metadata=document_metadata,
        )

        print(f"✅ Correction completed")
        print(f"🔧 Corrections applied: {corrections}")
        print(f"📝 Original length: {len(sample_ocr_text)} characters")
        print(f"📝 Corrected length: {len(corrected_text)} characters")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)

        # Save original and corrected text
        with open(
            output_dir / "context_aware_original.txt", "w", encoding="utf-8"
        ) as f:
            f.write(sample_ocr_text)

        with open(
            output_dir / "context_aware_corrected.txt", "w", encoding="utf-8"
        ) as f:
            f.write(corrected_text)

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Context-aware correction failed: {e}")


def demo_ollama_integration():
    """Demonstrate Ollama integration status."""
    print("\n🤖 Ollama Integration Status")
    print("=" * 50)

    try:
        import requests

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]

            print("✅ Ollama is running")
            print(f"📦 Available models: {available_models}")

            # Check for recommended models
            recommended_models = ["llama2", "codellama", "mistral"]
            installed_recommended = [
                model for model in recommended_models if model in available_models
            ]

            if installed_recommended:
                print(f"🎯 Recommended models installed: {installed_recommended}")
            else:
                print("⚠️  No recommended models found. Consider installing:")
                print("   - llama2: General purpose model")
                print("   - codellama: Code-focused model")
                print("   - mistral: Efficient model")
        else:
            print("❌ Ollama API not responding")

    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("💡 To start Ollama:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull a model: ollama pull llama2")
    except ImportError:
        print("❌ Requests library not available")


def main():
    """Main demo function."""
    print("🚀 jtext Phase 2 Features Demo")
    print("=" * 60)
    print("This demo showcases the new Phase 2 features:")
    print("- Enhanced LLM integration with Ollama")
    print("- Document processing (PDF, DOCX, PPTX, HTML)")
    print("- Audio transcription with Whisper")
    print("- Context-aware correction")
    print("=" * 60)

    # Setup logging
    setup_logging(verbose=True)

    # Run demos
    demo_ollama_integration()
    demo_enhanced_ocr()
    demo_document_processing()
    demo_audio_transcription()
    demo_context_aware_correction()

    print("\n🎉 Phase 2 Demo Complete!")
    print("=" * 60)
    print("📁 Check the 'demo_output' directory for results")
    print("🔧 All Phase 2 features are now available in the jtext CLI")
    print("💡 Use 'jtext --help' to see all available commands")


if __name__ == "__main__":
    main()
