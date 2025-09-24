"""
LLM-based OCR Correction Service.

Implements text-only LLM correction for fast and practical OCR enhancement.
Supports both Ollama integration for local LLM inference.

Features:
- Text-only correction (fast, practical)
- Fallback multimodal support
- Context-aware error correction
- Confidence scoring
- Structured logging with performance metrics

@author: jtext Development Team
@since: 1.0.0
@compliance: AGENTS.md, Clean Architecture
"""

import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from PIL import Image
import io

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from ..core import Result, Ok, Err
from ..domain import Language, Confidence
from .logging import get_logger, start_performance_tracking


@dataclass
class LLMCorrectionRequest:
    """Request for LLM-based OCR correction."""

    original_text: str
    confidence_score: float
    image_path: str
    language: Language
    model_name: Optional[str] = None
    correction_strength: float = 0.5


@dataclass
class LLMCorrectionResult:
    """Result of LLM-based OCR correction."""

    corrected_text: str
    confidence: Confidence
    corrections_made: List[Dict[str, str]]
    reasoning: str
    processing_time_ms: float
    model_used: str


class OllamaLLMCorrectionService:
    """
    Ollama-based LLM correction service.

    Implements text-only OCR correction using local Ollama deployment.
    Optimized for speed and practical use cases.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama LLM correction service."""
        self.base_url = base_url
        self.logger = get_logger("jtext.infrastructure.llm_correction")

        # Test Ollama connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            if not HAS_REQUESTS:
                self.logger.warn(
                    "Requests library not available, LLM correction disabled",
                    operation="connection_test",
                    component="ollama",
                    tags={"category": "system", "result": "dependency_missing"},
                )
                return False

            response = requests.get(f"{self.base_url}/api/tags", timeout=5)

            if response.status_code == 200:
                models = response.json().get("models", [])
                self.logger.info(
                    "Ollama connection successful",
                    operation="connection_test",
                    component="ollama",
                    available_models=len(models),
                    tags={"category": "system", "result": "success"},
                )
                return True
            else:
                self.logger.warn(
                    "Ollama server responded with error",
                    operation="connection_test",
                    component="ollama",
                    status_code=response.status_code,
                    tags={"category": "system", "result": "server_error"},
                )
                return False

        except Exception as e:
            self.logger.warn(
                "Failed to connect to Ollama server",
                operation="connection_test",
                component="ollama",
                error=str(e),
                tags={"category": "system", "result": "connection_failed"},
            )
            return False

    def correct_ocr_text(
        self,
        text: str,
        confidence: float,
        language: Language,
        model_name: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Result[LLMCorrectionResult, str]:
        """
        Correct OCR text using advanced LLM analysis with image context.

        Supports both text-only and multimodal correction based on OCR quality.
        For poor OCR results, uses image analysis to reconstruct content.

        Args:
            text: Original OCR text
            confidence: OCR confidence score
            language: Text language
            model_name: LLM model to use
            image_path: Optional image path for multimodal analysis

        Returns:
            Result containing corrected text or error message
        """
        metrics = start_performance_tracking()

        try:
            # Determine correction strategy based on OCR quality
            correction_strategy = self._determine_correction_strategy(
                text, confidence, image_path
            )

            self.logger.info(
                f"Starting LLM-OCR complement with {correction_strategy} strategy",
                operation="llm_ocr_complement",
                component="ollama",
                original_text_length=len(text),
                confidence_score=confidence,
                language=language.value,
                model=model_name or "default",
                strategy=correction_strategy,
                tags={"category": "processing", "operation": "llm_ocr_complement"},
            )

            # Always use multimodal prompt for LLM-OCR complement
            if correction_strategy == "multimodal" and image_path:
                prompt = self._generate_multimodal_prompt(
                    text, confidence, language, image_path
                )
            else:
                # If no image available, return error - we require image for LLM-OCR
                return Err(
                    "Image path required for LLM-OCR complement. Text-only correction is not supported."
                )

            # Select appropriate model
            model = self._select_model(model_name, language)

            # Always use multimodal API for LLM-OCR complement
            if correction_strategy == "multimodal" and image_path:
                result = self._call_ollama_multimodal_api(model, prompt, image_path)
            else:
                return Err("Multimodal processing required for LLM-OCR complement.")

            if result.is_err:
                return result

            # Parse and validate result
            parsed_result = self._parse_llm_response(
                result.unwrap(), text, confidence, model, metrics
            )

            if parsed_result.is_ok:
                correction = parsed_result.unwrap()

                self.logger.info(
                    "LLM-OCR complement completed successfully",
                    operation="llm_ocr_complement",
                    component="ollama",
                    original_length=len(text),
                    corrected_length=len(correction.corrected_text),
                    corrections_count=len(correction.corrections_made),
                    final_confidence=correction.confidence.to_float(),
                    processing_time_ms=correction.processing_time_ms,
                    tags={"category": "processing", "result": "success"},
                )

                return Ok(correction)
            else:
                return parsed_result

        except Exception as e:
            metrics.finish()
            error_msg = f"LLM correction failed: {str(e)}"

            self.logger.error(
                "LLM-OCR complement failed with exception",
                operation="llm_ocr_complement",
                component="ollama",
                error=str(e),
                processing_time_ms=metrics.duration_ms or 0.0,
                tags={"category": "system_error", "result": "failure"},
            )

            return Err(error_msg)

    def _determine_correction_strategy(
        self, text: str, confidence: float, image_path: Optional[str]
    ) -> str:
        """
        Determine the best correction strategy based on available resources.

        Args:
            text: OCR text to analyze
            confidence: OCR confidence score
            image_path: Optional image path for multimodal analysis

        Returns:
            Strategy: 'multimodal' if image available, otherwise 'enhanced_text'
        """
        # Always use multimodal if image is available
        if image_path:
            return "multimodal"

        # Fallback to enhanced text if no image
        return "enhanced_text"

    def _assess_text_quality(self, text: str) -> float:
        """
        Assess the quality of OCR text to determine correction strategy.

        Args:
            text: OCR text to assess

        Returns:
            Quality score between 0.0 (poor) and 1.0 (excellent)
        """
        if not text or len(text.strip()) < 2:
            return 0.0

        # Check for common OCR error patterns
        error_indicators = [
            # Random character sequences
            r"[^a-zA-Z0-9\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF.,!?@#$%^&*()_+-=]",
            # Excessive special characters
            r"[^a-zA-Z0-9\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]",
            # Repeated characters (likely OCR errors)
            r"(.)\1{3,}",
            # Mixed case inconsistencies
            r"[a-z][A-Z][a-z]",
        ]

        import re

        # Count error indicators
        error_count = 0
        for pattern in error_indicators:
            error_count += len(re.findall(pattern, text))

        # Calculate quality score
        text_length = len(text)
        if text_length == 0:
            return 0.0

        # Base score from length and error ratio
        base_score = min(1.0, text_length / 20)  # Longer text = higher base score
        error_penalty = min(0.8, error_count / text_length * 10)  # Error penalty

        quality_score = max(0.0, base_score - error_penalty)

        # Bonus for recognizable words
        word_bonus = 0
        words = text.split()
        for word in words:
            if len(word) > 2 and any(c.isalpha() for c in word):
                word_bonus += 0.1

        return min(1.0, quality_score + word_bonus)

    def _generate_enhanced_text_prompt(
        self, text: str, confidence: float, language: Language
    ) -> str:
        """Generate enhanced prompt for poor OCR text correction."""

        # Analyze text quality for specific instructions
        text_quality = self._assess_text_quality(text)

        if text_quality < 0.3:
            quality_instruction = "CRITICAL: The OCR text appears to be severely corrupted or meaningless. You must reconstruct the most likely intended content based on context clues, common patterns, and linguistic knowledge."
        elif text_quality < 0.6:
            quality_instruction = "WARNING: The OCR text has significant errors. Focus on correcting obvious mistakes and reconstructing missing or garbled words."
        else:
            quality_instruction = "The OCR text has minor errors. Make conservative corrections to improve readability."

        # Language-specific examples for poor OCR
        if language == Language.JAPANESE:
            examples = """
Examples of common OCR errors in Japanese text:
- "こんにちは" might be read as "こんにちほ" or "こんにちぱ"
- "テスト" might be read as "テストロ" or "テスk"
- English mixed with Japanese: "Hello World" + "こんにちは"
"""
        else:
            examples = """
Examples of common OCR errors in English text:
- "Hello World" might be read as "HelloWerd" or "Hello Wor1d"
- "Test Image" might be read as "Tpst Imagp" or "Test lmage"
- Numbers: "123" might be read as "l23" or "1Z3"
- "OCR often misreads" might be read as "OCRofen msreads"
- "Special chars" might be read as "SecgNchars"
"""

        # Confidence-based instruction
        if confidence < 0.3:
            confidence_instruction = (
                "The OCR confidence is very low. Please perform aggressive correction."
            )
        elif confidence < 0.7:
            confidence_instruction = (
                "The OCR confidence is moderate. Please correct obvious errors."
            )
        else:
            confidence_instruction = (
                "The OCR confidence is high. Please make only conservative corrections."
            )

        return f"""You are an expert OCR text correction specialist. Your task is to correct OCR errors in the given text.

CONTEXT:
- Language: {language.value}
- Original OCR Confidence: {confidence:.2%}
- Correction Strategy: {confidence_instruction}

{examples}

ORIGINAL OCR TEXT:
{text}

INSTRUCTIONS:
1. Analyze the OCR text for common recognition errors
2. Correct obvious mistakes while preserving the original structure
3. Pay attention to:
   - Character substitutions (similar-looking characters like 'l'↔'1', 'o'↔'0', 'rn'↔'m')
   - Word boundaries and spacing
   - Punctuation and special characters
   - Mixed language content

4. Provide your response in this exact JSON format:
{{
    "corrected_text": "The corrected text here",
    "confidence": 0.95,
    "corrections": [
        {{"original": "HelloWerd", "corrected": "Hello World", "reason": "OCR misread characters"}},
        {{"original": "OCRofen", "corrected": "OCR often", "reason": "Missing space and character error"}}
    ],
    "reasoning": "Brief explanation of correction approach and key changes made"
}}

Focus on accuracy. Only correct clear OCR errors that you can identify from context and common OCR mistakes."""

    def _generate_multimodal_prompt(
        self, text: str, confidence: float, language: Language, image_path: str
    ) -> str:
        """Generate multimodal prompt for image-based OCR correction."""

        # Analyze text quality for specific instructions
        text_quality = self._assess_text_quality(text)

        if text_quality < 0.3:
            quality_instruction = "CRITICAL: The OCR text is severely corrupted. Use the image to reconstruct the complete content, paying attention to layout, formatting, and visual context."
        elif text_quality < 0.6:
            quality_instruction = "WARNING: The OCR text has significant errors. Use the image to identify and correct mistakes, especially layout and formatting issues."
        else:
            quality_instruction = "The OCR text has minor errors. Use the image to make precise corrections and improve formatting."

        # Language-specific multimodal examples
        if language == Language.JAPANESE:
            examples = """
Examples of multimodal OCR correction:
- OCR: "こんにちは" → Image shows: "こんにちは" (correct)
- OCR: "テスト" → Image shows: "テスト" with proper formatting
- OCR: "Hello World" → Image shows: "Hello World" in proper layout
- OCR: garbled text → Image shows: clear text that can be reconstructed
"""
        else:
            examples = """
Examples of multimodal OCR correction:
- OCR: "Helo Wrld" → Image shows: "Hello World" (correct)
- OCR: "test" → Image shows: "Test" with proper capitalization
- OCR: garbled text → Image shows: clear text that can be reconstructed
- OCR: missing words → Image shows: complete text with proper spacing
"""

        return f"""You are an expert multimodal OCR correction specialist. Your task is to use the image to complement and improve the Tesseract OCR results.

{examples}

{quality_instruction}

CRITICAL INSTRUCTIONS:
1. Use the image as the primary source of truth for text content
2. Compare the Tesseract OCR text with what you see in the image
3. Identify missing text, incorrect characters, and layout issues
4. Reconstruct the complete text content based on visual analysis
5. Preserve the original formatting, spacing, and structure from the image
6. If Tesseract missed text or made errors, use the image to fill in the gaps

Tesseract OCR Result: "{text}"

Your task is to provide the complete, accurate text content as it appears in the image, using the Tesseract result as a starting point but not limiting yourself to it.

Please provide the corrected version in JSON format:
{{
    "corrected_text": "complete text content from image analysis",
    "confidence": 0.95,
    "corrections": ["list of corrections and additions made based on image"],
    "reasoning": "explanation of how image analysis complemented the OCR result"
}}"""

    def _select_model(self, requested_model: Optional[str], language: Language) -> str:
        """Select appropriate model for correction task."""
        if requested_model:
            # If user requested a multimodal model, use it directly
            # Ollama will handle text-only requests appropriately
            return requested_model

        # Default model selection
        return "gemma3:4b"  # Good general-purpose model

    def _call_ollama_api(self, model_name: str, prompt: str) -> Result[str, str]:
        """Call Ollama API for text inference."""
        try:
            if not HAS_REQUESTS:
                return Err("Requests library not available")

            # Prepare API request (text-only)
            api_request = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for factual correction
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }

            self.logger.debug(
                "Calling Ollama API for text correction",
                operation="ollama_api",
                component="ollama",
                model=model_name,
                prompt_length=len(prompt),
                tags={"category": "api", "operation": "llm_inference"},
            )

            # Make API call with extended timeout for complex models
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=api_request,
                timeout=300,  # Extended timeout for larger models like gemma3:4b
            )

            if response.status_code == 200:
                result = response.json()
                return Ok(result.get("response", ""))
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                self.logger.error(
                    "Ollama API call failed",
                    operation="ollama_api",
                    component="ollama",
                    status_code=response.status_code,
                    error=response.text,
                    tags={"category": "api_error"},
                )
                return Err(error_msg)

        except requests.exceptions.Timeout:
            return Err("Ollama API timeout - processing took too long")
        except requests.exceptions.ConnectionError:
            return Err("Cannot connect to Ollama server")
        except Exception as e:
            return Err(f"Ollama API call failed: {str(e)}")

    def _call_ollama_multimodal_api(
        self, model_name: str, prompt: str, image_path: str
    ) -> Result[str, str]:
        """Call Ollama API for multimodal inference with image."""
        try:
            if not HAS_REQUESTS:
                return Err("Requests library not available")

            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            if image_base64.is_err:
                return image_base64

            # Prepare API request (multimodal)
            api_request = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_base64.unwrap()],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent results
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }

            self.logger.debug(
                "Calling Ollama API for multimodal correction",
                operation="llm_multimodal_inference",
                component="ollama",
                model=model_name,
                prompt_length=len(prompt),
                image_path=image_path,
                tags={"category": "api", "operation": "llm_multimodal_inference"},
            )

            # Make API call with extended timeout for multimodal processing
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=api_request,
                timeout=300,  # Extended timeout for multimodal processing
            )

            if response.status_code == 200:
                result = response.json()
                return Ok(result.get("response", ""))
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                self.logger.error(
                    "Ollama multimodal API call failed",
                    operation="llm_multimodal_inference",
                    component="ollama",
                    status_code=response.status_code,
                    error=response.text,
                    tags={"category": "api_error"},
                )
                return Err(error_msg)

        except requests.exceptions.Timeout:
            return Err("Ollama API timeout - multimodal processing took too long")
        except requests.exceptions.ConnectionError:
            return Err("Ollama API connection failed - check if Ollama is running")
        except Exception as e:
            return Err(f"Failed to call Ollama multimodal API: {str(e)}")

    def _encode_image_to_base64(self, image_path: str) -> Result[str, str]:
        """Encode image to base64 for multimodal API."""
        try:
            import base64

            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode("utf-8")
                return Ok(base64_encoded)

        except FileNotFoundError:
            return Err(f"Image file not found: {image_path}")
        except Exception as e:
            return Err(f"Failed to encode image: {str(e)}")

    def _parse_llm_response(
        self,
        response: str,
        original_text: str,
        original_confidence: float,
        model_used: str,
        metrics,
    ) -> Result[LLMCorrectionResult, str]:
        """Parse and validate LLM response."""
        try:
            metrics.finish()

            # Try to extract JSON from response
            response = response.strip()

            # Find JSON block if wrapped in other text
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                # Fallback: treat entire response as corrected text
                return Ok(
                    LLMCorrectionResult(
                        corrected_text=response,
                        confidence=Confidence.from_float(
                            min(0.8, original_confidence + 0.1)
                        ),
                        corrections_made=[],
                        reasoning="LLM provided text without structured format",
                        processing_time_ms=metrics.duration_ms or 0.0,
                        model_used=model_used,
                    )
                )

            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)

            # Validate required fields
            corrected_text = parsed.get("corrected_text", original_text)
            confidence_val = parsed.get("confidence", original_confidence)
            corrections = parsed.get("corrections", [])
            reasoning = parsed.get("reasoning", "LLM correction applied")

            # Ensure confidence is valid
            confidence_val = max(0.0, min(1.0, float(confidence_val)))

            return Ok(
                LLMCorrectionResult(
                    corrected_text=corrected_text,
                    confidence=Confidence.from_float(confidence_val),
                    corrections_made=corrections,
                    reasoning=reasoning,
                    processing_time_ms=metrics.duration_ms or 0.0,
                    model_used=model_used,
                )
            )

        except json.JSONDecodeError as e:
            # Fallback to treating response as corrected text
            return Ok(
                LLMCorrectionResult(
                    corrected_text=response.strip(),
                    confidence=Confidence.from_float(
                        min(0.8, original_confidence + 0.1)
                    ),
                    corrections_made=[],
                    reasoning=f"LLM response parsing failed: {str(e)}",
                    processing_time_ms=metrics.duration_ms or 0.0,
                    model_used=model_used,
                )
            )
        except Exception as e:
            return Err(f"Failed to parse LLM response: {str(e)}")

    def is_available(self) -> bool:
        """Check if LLM correction service is available."""
        return HAS_REQUESTS and self._test_connection()


def get_llm_correction_service(
    base_url: str = "http://localhost:11434",
) -> OllamaLLMCorrectionService:
    """Get LLM correction service instance."""
    return OllamaLLMCorrectionService(base_url)
