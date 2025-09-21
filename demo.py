#!/usr/bin/env python3
"""
Demo script for jtext - Japanese Text Processing CLI System

This script demonstrates the basic functionality of the jtext system
by creating a test image and processing it through the OCR pipeline.
"""

import tempfile
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add the jtext package to the path
sys.path.insert(0, str(Path(__file__).parent))

from jtext.core.ocr_hybrid import HybridOCR
from jtext.utils.logging import setup_logging


def create_test_image(text: str = "これはテストです。", output_path: str = None) -> str:
    """
    Create a test image with Japanese text for OCR testing.

    Args:
        text: Japanese text to render
        output_path: Output file path (optional)

    Returns:
        Path to the created image file
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    # Create a white image
    width, height = 400, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Try to use a Japanese font, fallback to default
    try:
        # Try common Japanese fonts on macOS
        font_paths = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/System/Library/Fonts/Arial Unicode MS.ttf",
        ]

        font = None
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    font = ImageFont.truetype(font_path, 24)
                    break
                except:
                    continue

        if font is None:
            font = ImageFont.load_default()

    except:
        font = ImageFont.load_default()

    # Draw text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill="black", font=font)

    # Save image
    image.save(output_path)
    print(f"✅ Created test image: {output_path}")

    return output_path


def demo_ocr_processing():
    """Demonstrate OCR processing functionality."""
    print("🎯 jtext OCR Demo")
    print("=" * 50)

    # Setup logging
    setup_logging(verbose=True)

    # Create test image
    test_text = "これは日本語のテストです。OCR処理をテストしています。"
    image_path = create_test_image(test_text)

    try:
        # Initialize OCR processor
        print("\n🔧 Initializing OCR processor...")
        ocr = HybridOCR(enable_correction=False)

        # Process image
        print(f"\n📸 Processing image: {image_path}")
        result = ocr.process_image(image_path)

        # Display results
        print("\n📊 Results:")
        print(f"  Extracted text: {result.text}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Memory usage: {result.memory_usage:.1f}MB")

        # Test with correction enabled
        print("\n🔧 Testing with LLM correction...")
        ocr_with_correction = HybridOCR(enable_correction=True)
        result_corrected = ocr_with_correction.process_image(image_path)

        print("\n📊 Corrected Results:")
        print(f"  Extracted text: {result_corrected.text}")
        print(f"  Confidence: {result_corrected.confidence:.2f}")
        print(f"  Corrections applied: {result_corrected.corrections_applied}")
        print(f"  Correction ratio: {result_corrected.correction_ratio:.3f}")

        # Save results
        output_dir = Path("./demo_output")
        output_dir.mkdir(exist_ok=True)

        from jtext.utils.io_utils import save_results

        save_results(result, output_dir / "demo_result")
        save_results(result_corrected, output_dir / "demo_result_corrected")

        print(f"\n💾 Results saved to: {output_dir}")
        print("   - demo_result.txt / demo_result.json")
        print("   - demo_result_corrected.txt / demo_result_corrected.json")

    except Exception as e:
        print(f"❌ Error during OCR processing: {e}")
        return False

    finally:
        # Cleanup
        if Path(image_path).exists():
            Path(image_path).unlink()
            print(f"\n🧹 Cleaned up test image: {image_path}")

    return True


def demo_cli_interface():
    """Demonstrate CLI interface functionality."""
    print("\n🖥️  CLI Interface Demo")
    print("=" * 50)

    # Create test image
    test_text = "CLIテスト用の画像です。"
    image_path = create_test_image(test_text)

    try:
        import subprocess

        # Test CLI help
        print("\n📖 Testing CLI help...")
        result = subprocess.run(
            ["python", "-m", "jtext.cli", "--help"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ CLI help works")
        else:
            print(f"❌ CLI help failed: {result.stderr}")

        # Test OCR command
        print(f"\n🔧 Testing OCR command with: {image_path}")
        result = subprocess.run(
            ["python", "-m", "jtext.cli", "ocr", image_path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ OCR command works")
            print(f"Output: {result.stdout}")
        else:
            print(f"❌ OCR command failed: {result.stderr}")

    except Exception as e:
        print(f"❌ Error during CLI demo: {e}")
        return False

    finally:
        # Cleanup
        if Path(image_path).exists():
            Path(image_path).unlink()

    return True


def main():
    """Main demo function."""
    print("🚀 jtext - Japanese Text Processing CLI System Demo")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("jtext").exists():
        print("❌ Error: jtext package not found")
        print("   Please run this script from the project root directory")
        return 1

    # Run demos
    success = True

    try:
        # OCR processing demo
        if not demo_ocr_processing():
            success = False

        # CLI interface demo
        if not demo_cli_interface():
            success = False

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Install system dependencies: ./install.sh")
        print("   2. Activate virtual environment: source venv/bin/activate")
        print("   3. Test with your own images: jtext ocr your_image.png")
        return 0
    else:
        print("\n⚠️  Demo completed with some issues")
        print("   Check the error messages above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
