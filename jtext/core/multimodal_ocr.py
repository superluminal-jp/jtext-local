"""
Multimodal OCR processing with image understanding and text extraction.

This module implements advanced OCR functionality that combines:
1. Traditional OCR (Tesseract) for text extraction
2. Vision-Language Models for image understanding
3. LLM-based fusion for optimal results
"""

import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pytesseract
from PIL import Image
import psutil
import requests
import json

from ..utils.logging import get_logger
from ..preprocessing.image_prep import ImagePreprocessor
from ..correction.ocr_corrector import OCRCorrector
from ..correction.context_aware_corrector import ContextAwareCorrector
from ..utils.validation import validate_image_file


logger = get_logger(__name__)


class MultimodalOCRResult:
    """Container for multimodal OCR processing results with enhanced metadata."""

    def __init__(
        self,
        source_path: str,
        text: str,
        confidence: float,
        processing_time: float,
        memory_usage: float,
        corrections_applied: int = 0,
        correction_ratio: float = 0.0,
        vision_analysis: Optional[Dict[str, Any]] = None,
        fusion_method: str = "ocr_only",
    ):
        self.source_path = source_path
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.memory_usage = memory_usage
        self.corrections_applied = corrections_applied
        self.correction_ratio = correction_ratio
        self.vision_analysis = vision_analysis or {}
        self.fusion_method = fusion_method
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "source": str(Path(self.source_path).absolute()),
            "type": "image",
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
            ),
            "processing": {
                "pipeline": self._get_pipeline(),
                "fusion_method": self.fusion_method,
                "ocr_engine": "tesseract-5.3.3",
                "vision_model": self.vision_analysis.get("model", None),
                "llm_model": "gpt-oss" if self.corrections_applied > 0 else None,
                "confidence": {
                    "ocr_raw": self.confidence,
                    "vision_analysis": self.vision_analysis.get("confidence", None),
                    "llm_corrected": (
                        self.confidence if self.corrections_applied > 0 else None
                    ),
                },
            },
            "vision_analysis": self.vision_analysis,
            "correction_stats": {
                "characters_changed": self.corrections_applied,
                "words_changed": self.corrections_applied // 2,
                "correction_ratio": self.correction_ratio,
                "correction_types": (
                    ["kanji_fix", "punctuation", "layout", "vision_enhanced"]
                    if self.corrections_applied > 0
                    else []
                ),
            },
            "quality_metrics": {
                "character_count": len(self.text),
                "word_count": len(self.text.split()),
                "processing_time_sec": self.processing_time,
                "memory_usage_mb": self.memory_usage,
            },
        }

    def _get_pipeline(self) -> List[str]:
        """Get processing pipeline based on enabled features."""
        pipeline = ["tesseract"]
        if self.vision_analysis:
            pipeline.append("vision_analysis")
        if self.corrections_applied > 0:
            pipeline.append("llm_correction")
        return pipeline


class MultimodalOCR:
    """
    Advanced OCR processing combining traditional OCR with vision-language models.

    This class provides multimodal OCR functionality that:
    1. Extracts text using Tesseract OCR
    2. Analyzes image content using vision models
    3. Fuses results using LLM for optimal accuracy
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        enable_correction: bool = False,
        enable_vision: bool = False,
    ):
        """
        Initialize the multimodal OCR processor.

        Args:
            llm_model: LLM model name for correction and fusion
            vision_model: Vision model name for image analysis
            enable_correction: Whether to enable LLM correction
            enable_vision: Whether to enable vision analysis
        """
        self.enable_correction = enable_correction
        self.enable_vision = enable_vision
        self.llm_model = llm_model
        self.vision_model = vision_model or "llava"
        self.preprocessor = ImagePreprocessor()
        self.corrector = OCRCorrector(model=llm_model) if enable_correction else None
        self.context_corrector = (
            ContextAwareCorrector(model=llm_model) if enable_correction else None
        )

        logger.info(
            f"Initialized MultimodalOCR - Vision: {enable_vision}, "
            f"Correction: {enable_correction}"
        )

    def process_image(self, image_path: str) -> MultimodalOCRResult:
        """
        Process image through multimodal OCR pipeline.

        Args:
            image_path: Path to the image file

        Returns:
            MultimodalOCRResult with enhanced text and metadata

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image file is invalid
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        logger.info(f"Starting multimodal OCR processing for: {image_path}")

        # Validate input file
        if not validate_image_file(image_path):
            raise ValueError(f"Invalid image file: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocessor.preprocess(image)

            # Step 1: Traditional OCR
            ocr_text, ocr_confidence = self._extract_text_ocr(processed_image)

            # Step 2: Vision analysis (if enabled)
            vision_analysis = None
            if self.enable_vision:
                vision_analysis = self._analyze_image_vision(image_path)

            # Step 3: Fusion and correction
            final_text, corrections_applied, correction_ratio, fusion_method = (
                self._fuse_and_correct(ocr_text, vision_analysis, image_path)
            )

            # Calculate processing metrics
            processing_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory

            result = MultimodalOCRResult(
                source_path=image_path,
                text=final_text,
                confidence=ocr_confidence,
                processing_time=processing_time,
                memory_usage=memory_usage,
                corrections_applied=corrections_applied,
                correction_ratio=correction_ratio,
                vision_analysis=vision_analysis,
                fusion_method=fusion_method,
            )

            logger.info(
                f"Multimodal OCR completed in {processing_time:.2f}s, "
                f"confidence: {ocr_confidence:.2f}, corrections: {corrections_applied}, "
                f"fusion: {fusion_method}"
            )

            return result

        except Exception as e:
            logger.error(f"Multimodal OCR processing failed for {image_path}: {e}")
            raise

    def _extract_text_ocr(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text using traditional OCR."""
        logger.debug("Running Tesseract OCR")
        ocr_text = pytesseract.image_to_string(
            image,
            lang="jpn+eng",
            config="--psm 6",  # Uniform block of text
        )

        # Get confidence score
        ocr_data = pytesseract.image_to_data(
            image, lang="jpn+eng", output_type=pytesseract.Output.DICT
        )
        confidence = self._calculate_confidence(ocr_data)

        return ocr_text.strip(), confidence

    def _analyze_image_vision(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Analyze image using vision-language model."""
        if not self.enable_vision:
            return None

        logger.debug("Running vision analysis")
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # Prepare vision analysis prompt
            vision_prompt = self._create_vision_prompt()

            # Call Ollama with vision model
            payload = {
                "model": self.vision_model,
                "prompt": vision_prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                },
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                vision_text = result.get("response", "").strip()

                return {
                    "model": self.vision_model,
                    "analysis": vision_text,
                    "confidence": 0.8,  # Vision models typically have high confidence
                    "timestamp": time.time(),
                }
            else:
                logger.warning(f"Vision analysis failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return None

    def _create_vision_prompt(self) -> str:
        """Create prompt for vision analysis."""
        return """
この画像を詳しく分析し、以下の情報を提供してください：

1. 画像の内容の説明
2. 含まれているテキスト（可能な限り正確に）
3. 文書の種類（技術文書、手紙、表、図表など）
4. レイアウトの特徴
5. 読み取りにくい部分や注意点

日本語で回答してください。
"""

    def _fuse_and_correct(
        self,
        ocr_text: str,
        vision_analysis: Optional[Dict[str, Any]],
        image_path: str,
    ) -> Tuple[str, int, float, str]:
        """Fuse OCR and vision results with LLM correction."""
        if not self.enable_correction or not self.corrector:
            return ocr_text, 0, 0.0, "ocr_only"

        # Determine fusion strategy
        if vision_analysis:
            return self._fuse_with_vision(ocr_text, vision_analysis, image_path)
        else:
            return self._correct_ocr_only(ocr_text)

    def _fuse_with_vision(
        self, ocr_text: str, vision_analysis: Dict[str, Any], image_path: str
    ) -> Tuple[str, int, float, str]:
        """Fuse OCR and vision results using multimodal model for comprehensive analysis."""
        if not self.context_corrector:
            return self._correct_ocr_only(ocr_text)

        # Step 4: Multimodal fusion with comprehensive analysis
        try:
            # Use multimodal model for comprehensive text correction
            corrected_text, corrections_applied = self._multimodal_fusion_correction(
                ocr_text, vision_analysis, image_path
            )
            
            correction_ratio = corrections_applied / len(ocr_text) if ocr_text else 0.0
            return corrected_text, corrections_applied, correction_ratio, "multimodal_fusion"
            
        except Exception as e:
            logger.warning(f"Multimodal fusion failed, falling back to context-aware correction: {e}")
            # Fallback to context-aware correction
            return self._context_aware_fusion_fallback(ocr_text, vision_analysis)

    def _correct_ocr_only(self, ocr_text: str) -> Tuple[str, int, float, str]:
        """Apply standard OCR correction."""
        if not self.corrector:
            return ocr_text, 0, 0.0, "ocr_only"

        corrected_text, corrections_applied = self.corrector.correct(ocr_text)
        correction_ratio = corrections_applied / len(ocr_text) if ocr_text else 0.0

        return corrected_text, corrections_applied, correction_ratio, "ocr_correction"

    def _multimodal_fusion_correction(
        self, ocr_text: str, vision_analysis: Dict[str, Any], image_path: str
    ) -> Tuple[str, int]:
        """
        Perform comprehensive multimodal fusion correction using vision model.
        
        This method combines:
        1. OCR extracted text
        2. Vision analysis results
        3. Original image
        4. Contextual metadata
        
        To produce the most accurate and coherent text output.
        """
        logger.debug("Starting multimodal fusion correction")
        
        try:
            # Encode image to base64 for multimodal input
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # Create comprehensive multimodal prompt
            multimodal_prompt = self._create_multimodal_fusion_prompt(
                ocr_text, vision_analysis
            )

            # Prepare multimodal payload
            payload = {
                "model": self.vision_model,
                "prompt": multimodal_prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.05,  # Very low temperature for consistency
                    "top_p": 0.85,
                    "max_tokens": 2048,
                    "repeat_penalty": 1.1,
                },
            }

            logger.debug("Calling multimodal fusion model")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result.get("response", "").strip()
                
                # Clean up the response
                corrected_text = self._clean_multimodal_response(corrected_text, ocr_text)
                
                # Calculate corrections
                corrections_applied = self._calculate_corrections(ocr_text, corrected_text)
                
                logger.info(f"Multimodal fusion completed: {corrections_applied} corrections applied")
                return corrected_text, corrections_applied
            else:
                logger.error(f"Multimodal fusion failed: {response.status_code}")
                raise Exception(f"API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Multimodal fusion error: {e}")
            raise

    def _create_multimodal_fusion_prompt(
        self, ocr_text: str, vision_analysis: Dict[str, Any]
    ) -> str:
        """
        Create comprehensive multimodal fusion prompt following best practices.
        
        This prompt combines all available information to produce the most accurate result.
        """
        vision_analysis_text = vision_analysis.get("analysis", "")
        document_type = self._extract_document_type(vision_analysis)
        layout_info = self._extract_layout_info(vision_analysis)
        
        # Build context information
        context_info = []
        if document_type != "general":
            context_info.append(f"文書タイプ: {document_type}")
        
        if layout_info.get("has_tables"):
            context_info.append("表が含まれています")
        if layout_info.get("has_lists"):
            context_info.append("リストが含まれています")
        if layout_info.get("has_images"):
            context_info.append("画像が含まれています")
        
        context_str = "、".join(context_info) if context_info else "一般的な文書"
        
        return f"""あなたは高精度なOCR修正専門家です。以下の情報を総合的に分析し、最も正確で自然な日本語テキストを出力してください。

## 分析対象情報

### 1. OCR抽出テキスト（修正対象）
```
{ocr_text}
```

### 2. 画像分析結果
{vision_analysis_text}

### 3. 文書コンテキスト
- {context_str}
- レイアウトタイプ: {layout_info.get('layout_type', 'freeform')}

## 修正方針

1. **文字認識の修正**
   - 明らかな誤認識文字を正確に修正
   - 文脈に合わない漢字を適切に修正
   - 専門用語の正確性を保持

2. **文書構造の保持**
   - 表の構造とセルの境界を正確に保持
   - リストの階層とインデントを維持
   - 段落の区切りを適切に配置

3. **文脈の整合性**
   - 画像分析結果とOCR結果の整合性を確保
   - 文書タイプに応じた文体の統一
   - 専門用語の一貫性を保持

4. **自然性の向上**
   - 日本語として自然な表現に修正
   - 句読点と改行を適切に配置
   - 読みやすさを重視

## 出力要件

- 修正されたテキストのみを出力してください
- 説明や注釈は含めないでください
- 元の構造と意味を保持してください
- 日本語として自然で正確な文章にしてください

修正版テキスト:"""

    def _clean_multimodal_response(self, response: str, original_text: str) -> str:
        """Clean up multimodal model response."""
        # Remove common prefixes
        prefixes_to_remove = [
            "修正版テキスト:",
            "修正されたテキスト:",
            "修正版:",
            "修正結果:",
            "出力:",
            "結果:",
        ]
        
        cleaned = response.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove any remaining prompt artifacts
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines that look like prompts or instructions
            if not any(keyword in line for keyword in [
                "修正", "分析", "文書", "画像", "OCR", "テキスト", "結果", "出力"
            ]) or len(line.strip()) > 10:
                filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines).strip()
        
        # If result is too different from original, return original
        if len(result) < len(original_text) * 0.5:
            logger.warning("Multimodal response seems too short, using original text")
            return original_text
            
        return result

    def _calculate_corrections(self, original: str, corrected: str) -> int:
        """Calculate number of corrections made."""
        if not original or not corrected:
            return 0
            
        # Simple character-level difference calculation
        # This could be enhanced with more sophisticated diff algorithms
        original_chars = list(original)
        corrected_chars = list(corrected)
        
        # Use Levenshtein distance approximation
        max_len = max(len(original_chars), len(corrected_chars))
        min_len = min(len(original_chars), len(corrected_chars))
        
        # Count character differences
        differences = 0
        for i in range(min_len):
            if original_chars[i] != corrected_chars[i]:
                differences += 1
        
        # Add length difference
        differences += max_len - min_len
        
        return differences

    def _context_aware_fusion_fallback(
        self, ocr_text: str, vision_analysis: Dict[str, Any]
    ) -> Tuple[str, int, float, str]:
        """Fallback to context-aware correction when multimodal fusion fails."""
        logger.debug("Using context-aware correction fallback")
        
        # Create document metadata from vision analysis
        document_metadata = {
            "format": "image",
            "vision_analysis": vision_analysis.get("analysis", ""),
            "document_type": self._extract_document_type(vision_analysis),
            "layout_info": self._extract_layout_info(vision_analysis),
        }

        # Use context-aware correction with vision context
        corrected_text, corrections_applied = (
            self.context_corrector.correct_with_context(
                text=ocr_text,
                context_type="vision_enhanced",
                document_metadata=document_metadata,
            )
        )

        correction_ratio = corrections_applied / len(ocr_text) if ocr_text else 0.0
        return corrected_text, corrections_applied, correction_ratio, "context_aware_fallback"

    def _extract_document_type(self, vision_analysis: Dict[str, Any]) -> str:
        """Extract document type from vision analysis."""
        analysis = vision_analysis.get("analysis", "").lower()

        if "技術" in analysis or "technical" in analysis:
            return "technical"
        elif "学術" in analysis or "論文" in analysis:
            return "academic"
        elif "ビジネス" in analysis or "business" in analysis:
            return "business"
        elif "表" in analysis or "table" in analysis:
            return "tabular"
        else:
            return "general"

    def _extract_layout_info(self, vision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract layout information from vision analysis."""
        analysis = vision_analysis.get("analysis", "")

        return {
            "has_tables": "表" in analysis or "table" in analysis,
            "has_images": "画像" in analysis or "image" in analysis,
            "has_lists": "リスト" in analysis or "list" in analysis,
            "layout_type": "structured" if "構造" in analysis else "freeform",
        }

    def _calculate_confidence(self, ocr_data: Dict[str, Any]) -> float:
        """Calculate average confidence score from Tesseract output."""
        confidences = [int(conf) for conf in ocr_data["conf"] if int(conf) > 0]
        return sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

    def get_supported_vision_models(self) -> List[str]:
        """Get list of supported vision models."""
        return ["llava", "llava:7b", "llava:13b", "bakllava"]

    def check_vision_model_availability(self) -> bool:
        """Check if vision model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.vision_model in available_models
            return False
        except Exception:
            return False
