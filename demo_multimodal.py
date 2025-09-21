"""
Demo script for multimodal OCR functionality.

This script demonstrates the advanced multimodal OCR capabilities
that combine traditional OCR with vision-language models for optimal results.
"""

import click
from pathlib import Path
import tempfile
import shutil
import os
import requests
import json

from jtext.core.multimodal_ocr import MultimodalOCR
from jtext.utils.io_utils import ensure_output_dir, save_results
from jtext.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def check_ollama_status():
    """Check Ollama server status and available models."""
    print("\n🤖 Ollama Status Check")
    print("=" * 50)

    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]

            print("✅ Ollama is running")
            print(f"📦 Available models: {available_models}")

            # Check for vision models
            vision_models = ["llava", "llava:7b", "llava:13b", "bakllava"]
            available_vision = [model for model in vision_models if model in available_models]

            if available_vision:
                print(f"👁️  Vision models available: {available_vision}")
                return True, available_vision
            else:
                print("⚠️  No vision models found")
                print("💡 To install a vision model:")
                print("   ollama pull llava")
                return True, []
        else:
            print("❌ Ollama API not responding")
            return False, []

    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("💡 To start Ollama:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull a vision model: ollama pull llava")
        return False, []


def demo_ocr_only():
    """Demonstrate traditional OCR processing."""
    print("\n🔍 Traditional OCR Demo")
    print("=" * 50)

    # Initialize OCR processor (OCR only)
    ocr_processor = MultimodalOCR(
        enable_correction=False,
        enable_vision=False
    )

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
        print(f"🔧 Fusion method: {result.fusion_method}")
        print(f"📝 Text length: {len(result.text)} characters")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        save_results(result, output_dir / "ocr_only")

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ OCR processing failed: {e}")


def demo_ocr_with_correction():
    """Demonstrate OCR with LLM correction."""
    print("\n🔧 OCR with LLM Correction Demo")
    print("=" * 50)

    # Initialize OCR processor with correction
    ocr_processor = MultimodalOCR(
        llm_model="llama2",
        enable_correction=True,
        enable_vision=False
    )

    # Check if we have a sample image
    sample_image = Path("sample_image.png")
    if not sample_image.exists():
        print("⚠️  No sample image found. Please add a sample_image.png file.")
        return

    try:
        print(f"Processing image: {sample_image}")
        result = ocr_processor.process_image(str(sample_image))

        print(f"✅ OCR with correction completed in {result.processing_time:.2f}s")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"🔧 Corrections applied: {result.corrections_applied}")
        print(f"📈 Correction ratio: {result.correction_ratio:.2f}")
        print(f"🔧 Fusion method: {result.fusion_method}")
        print(f"📝 Text length: {len(result.text)} characters")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        save_results(result, output_dir / "ocr_corrected")

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ OCR with correction failed: {e}")


def demo_vision_analysis():
    """Demonstrate vision analysis capabilities."""
    print("\n👁️  Vision Analysis Demo")
    print("=" * 50)

    # Check Ollama status
    ollama_running, vision_models = check_ollama_status()
    if not ollama_running or not vision_models:
        print("❌ Ollama or vision models not available")
        return

    # Initialize OCR processor with vision
    ocr_processor = MultimodalOCR(
        vision_model=vision_models[0],
        enable_vision=True,
        enable_correction=False
    )

    # Check if we have a sample image
    sample_image = Path("sample_image.png")
    if not sample_image.exists():
        print("⚠️  No sample image found. Please add a sample_image.png file.")
        return

    try:
        print(f"Processing image with vision analysis: {sample_image}")
        result = ocr_processor.process_image(str(sample_image))

        print(f"✅ Vision analysis completed in {result.processing_time:.2f}s")
        print(f"👁️  Vision model: {result.vision_analysis.get('model', 'N/A')}")
        print(f"📊 Vision confidence: {result.vision_analysis.get('confidence', 'N/A')}")
        print(f"🔧 Fusion method: {result.fusion_method}")
        print(f"📝 Text length: {len(result.text)} characters")

        # Show vision analysis
        if result.vision_analysis:
            print(f"\n🔍 Vision Analysis:")
            print(f"   {result.vision_analysis.get('analysis', 'N/A')}")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        save_results(result, output_dir / "vision_analysis")

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Vision analysis failed: {e}")


def demo_full_multimodal():
    """Demonstrate full multimodal OCR processing."""
    print("\n🚀 Full Multimodal OCR Demo")
    print("=" * 50)

    # Check Ollama status
    ollama_running, vision_models = check_ollama_status()
    if not ollama_running or not vision_models:
        print("❌ Ollama or vision models not available")
        return

    # Initialize full multimodal OCR processor
    ocr_processor = MultimodalOCR(
        llm_model="llama2",
        vision_model=vision_models[0],
        enable_correction=True,
        enable_vision=True
    )

    # Check if we have a sample image
    sample_image = Path("sample_image.png")
    if not sample_image.exists():
        print("⚠️  No sample image found. Please add a sample_image.png file.")
        return

    try:
        print(f"Processing image with full multimodal OCR: {sample_image}")
        result = ocr_processor.process_image(str(sample_image))

        print(f"✅ Full multimodal OCR completed in {result.processing_time:.2f}s")
        print(f"📊 OCR Confidence: {result.confidence:.2f}")
        print(f"👁️  Vision model: {result.vision_analysis.get('model', 'N/A')}")
        print(f"📊 Vision confidence: {result.vision_analysis.get('confidence', 'N/A')}")
        print(f"🔧 Corrections applied: {result.corrections_applied}")
        print(f"📈 Correction ratio: {result.correction_ratio:.2f}")
        print(f"🔧 Fusion method: {result.fusion_method}")
        print(f"📝 Text length: {len(result.text)} characters")

        # Show processing pipeline
        pipeline = result.to_dict()["processing"]["pipeline"]
        print(f"🔄 Processing pipeline: {' → '.join(pipeline)}")

        # Show vision analysis
        if result.vision_analysis:
            print(f"\n🔍 Vision Analysis:")
            print(f"   {result.vision_analysis.get('analysis', 'N/A')}")

        # Save results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        save_results(result, output_dir / "multimodal_ocr")

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Full multimodal OCR failed: {e}")


def demo_comparison():
    """Demonstrate comparison between different OCR methods."""
    print("\n📊 OCR Methods Comparison Demo")
    print("=" * 50)

    # Check if we have a sample image
    sample_image = Path("sample_image.png")
    if not sample_image.exists():
        print("⚠️  No sample image found. Please add a sample_image.png file.")
        return

    methods = [
        ("Traditional OCR", MultimodalOCR(enable_correction=False, enable_vision=False)),
        ("OCR + LLM Correction", MultimodalOCR(llm_model="llama2", enable_correction=True, enable_vision=False)),
        ("OCR + Vision Analysis", MultimodalOCR(vision_model="llava", enable_vision=True, enable_correction=False)),
    ]

    # Check for vision models
    ollama_running, vision_models = check_ollama_status()
    if ollama_running and vision_models:
        methods.append(
            ("Full Multimodal", MultimodalOCR(
                llm_model="llama2",
                vision_model=vision_models[0],
                enable_correction=True,
                enable_vision=True
            ))
        )

    results = []

    for method_name, processor in methods:
        try:
            print(f"\n🔍 Testing: {method_name}")
            result = processor.process_image(str(sample_image))

            results.append({
                "method": method_name,
                "text_length": len(result.text),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "corrections": getattr(result, 'corrections_applied', 0),
                "fusion_method": getattr(result, 'fusion_method', 'N/A'),
                "text": result.text[:100] + "..." if len(result.text) > 100 else result.text
            })

            print(f"   ✅ Completed in {result.processing_time:.2f}s")
            print(f"   📊 Confidence: {result.confidence:.2f}")
            print(f"   📝 Text length: {len(result.text)} characters")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Show comparison table
    if results:
        print(f"\n📊 Comparison Results:")
        print("=" * 80)
        print(f"{'Method':<20} {'Time(s)':<8} {'Confidence':<10} {'Length':<8} {'Corrections':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['method']:<20} {result['processing_time']:<8.2f} "
                  f"{result['confidence']:<10.2f} {result['text_length']:<8} "
                  f"{result['corrections']:<12}")

        # Save comparison results
        output_dir = Path("demo_output")
        ensure_output_dir(output_dir)
        
        with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Comparison results saved to: {output_dir}/comparison_results.json")


def main():
    """Main function to run all multimodal OCR demos."""
    print("🚀 jtext Multimodal OCR Demo")
    print("=" * 60)
    print("This demo showcases the advanced multimodal OCR capabilities:")
    print("- Traditional OCR processing")
    print("- OCR with LLM correction")
    print("- Vision analysis with image understanding")
    print("- Full multimodal OCR (OCR + Vision + LLM)")
    print("- Performance comparison between methods")
    print("=" * 60)

    # Setup logging
    setup_logging(verbose=True)

    # Run demos
    demo_ocr_only()
    demo_ocr_with_correction()
    demo_vision_analysis()
    demo_full_multimodal()
    demo_comparison()

    print("\n🎉 Multimodal OCR Demo Complete!")
    print("=" * 60)
    print("📁 Check the 'demo_output' directory for results")
    print("🔧 Multimodal OCR features are now available in the jtext CLI")
    print("💡 Use 'jtext ocr --vision --llm-correct' for full multimodal processing")


if __name__ == "__main__":
    main()
