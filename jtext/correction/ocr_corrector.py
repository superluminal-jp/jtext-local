"""
LLM-based OCR correction for improved text accuracy.

This module provides LLM integration for correcting OCR results,
particularly for Japanese text where character recognition errors
are common.
"""

from typing import Tuple, Optional
import re
import json
import requests
import time

from ..utils.logging import get_logger


logger = get_logger(__name__)


class OCRCorrector:
    """
    LLM-based OCR result correction.

    This class provides correction functionality for OCR results using
    LLM models to improve accuracy and naturalness of extracted text.
    """

    CORRECTION_PROMPT = """
以下のテキストはOCRで抽出されたものです。
日本語として自然になるよう修正してください。

修正方針：
1. 明らかな誤認識文字を修正
2. 文脈に合わない漢字を適切に修正
3. 句読点や改行を適切に配置
4. 表やリストの構造を復元
5. 元の意味を変えない範囲で修正

元テキスト：{ocr_text}
修正版：
"""

    def __init__(self, model: str = "gpt-oss"):
        """
        Initialize the OCR corrector.

        Args:
            model: LLM model name to use for correction
        """
        self.model = model
        self._model_available = self._check_model_availability()

        if not self._model_available:
            logger.warning(
                f"LLM model {model} not available, using rule-based correction"
            )

    def correct(self, ocr_text: str) -> Tuple[str, int]:
        """
        Correct OCR text using LLM or rule-based methods.

        Args:
            ocr_text: Raw OCR text to correct

        Returns:
            Tuple of (corrected_text, number_of_corrections)
        """
        if not ocr_text.strip():
            return ocr_text, 0

        logger.debug(f"Correcting OCR text: {len(ocr_text)} characters")

        if self._model_available:
            corrected_text = self._llm_correct(ocr_text)
        else:
            corrected_text = self._rule_based_correct(ocr_text)

        # Count corrections made
        corrections = self._count_corrections(ocr_text, corrected_text)

        logger.debug(f"Applied {corrections} corrections")
        return corrected_text, corrections

    def _check_model_availability(self) -> bool:
        """
        Check if the specified LLM model is available.

        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]

                # Check if our model is available
                if self.model in available_models:
                    logger.info(f"LLM model {self.model} is available")
                    return True
                else:
                    logger.warning(
                        f"Model {self.model} not found. Available models: {available_models}"
                    )
                    return False
            else:
                logger.warning("Ollama API not responding")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def _llm_correct(self, ocr_text: str) -> str:
        """
        Correct text using LLM model.

        Args:
            ocr_text: Raw OCR text

        Returns:
            LLM-corrected text
        """
        try:
            prompt = self.CORRECTION_PROMPT.format(ocr_text=ocr_text)

            # Prepare request payload for Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent correction
                    "top_p": 0.9,
                    "max_tokens": 2048,
                },
            }

            logger.debug(f"Calling Ollama API with model: {self.model}")

            # Make request to Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result.get("response", "").strip()

                # Remove the prompt from the response if it was included
                if corrected_text.startswith(prompt):
                    corrected_text = corrected_text[len(prompt) :].strip()

                logger.info(
                    f"LLM correction completed: {len(corrected_text)} characters"
                )
                return corrected_text
            else:
                logger.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return self._rule_based_correct(ocr_text)

        except requests.exceptions.Timeout:
            logger.error("Ollama API timeout, falling back to rule-based correction")
            return self._rule_based_correct(ocr_text)
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Ollama API error: {e}, falling back to rule-based correction"
            )
            return self._rule_based_correct(ocr_text)
        except Exception as e:
            logger.error(
                f"Unexpected error in LLM correction: {e}, falling back to rule-based correction"
            )
            return self._rule_based_correct(ocr_text)

    def _rule_based_correct(self, ocr_text: str) -> str:
        """
        Apply rule-based corrections to OCR text.

        Args:
            ocr_text: Raw OCR text

        Returns:
            Rule-based corrected text
        """
        corrected = ocr_text

        # Common OCR error corrections for Japanese
        corrections = {
            # Common character misrecognitions
            "０": "0",
            "１": "1",
            "２": "2",
            "３": "3",
            "４": "4",
            "５": "5",
            "６": "6",
            "７": "7",
            "８": "8",
            "９": "9",
            # Common punctuation issues
            "，": "、",
            "．": "。",
            "：": "：",
            "；": "；",
            # Common spacing issues
            "　　": "　",
            "  ": " ",  # Multiple spaces to single
        }

        # Apply corrections
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)

        # Fix common spacing around punctuation
        corrected = re.sub(
            r"([。、])\s+", r"\1", corrected
        )  # Remove spaces after punctuation
        corrected = re.sub(
            r"\s+([。、])", r"\1", corrected
        )  # Remove spaces before punctuation

        # Normalize line breaks
        corrected = re.sub(r"\r\n|\r", "\n", corrected)
        corrected = re.sub(
            r"\n\s*\n\s*\n", "\n\n", corrected
        )  # Multiple empty lines to double

        return corrected

    def _count_corrections(self, original: str, corrected: str) -> int:
        """
        Count the number of corrections made.

        Args:
            original: Original OCR text
            corrected: Corrected text

        Returns:
            Number of corrections made
        """
        if len(original) != len(corrected):
            # If lengths differ, estimate based on character differences
            return abs(len(original) - len(corrected)) + sum(
                1 for a, b in zip(original, corrected) if a != b
            )

        # Count character-by-character differences
        return sum(1 for a, b in zip(original, corrected) if a != b)
