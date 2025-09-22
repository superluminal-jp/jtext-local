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

    CORRECTION_PROMPT = """あなたは日本語OCR修正の専門家です。以下のOCR抽出テキストを、日本語として自然で正確な文章に修正してください。

# OCR修正の基本原則

## 1. 文字認識エラーの修正
### 一般的なOCRエラーパターン：
- ひらがな⇔カタカナの誤認識（は⇔ハ、る⇔ル など）
- 似た形の漢字の誤認識（千⇔干、土⇔士、末⇔未 など）
- 英数字の誤認識（0⇔O、1⇔l、B⇔8 など）
- 句読点の欠落・誤配置（。、！？の位置）

### 修正アプローチ：
- 前後の文脈から正しい文字を推定
- 一般的な日本語表現との照合
- 専門用語・固有名詞の正確性確保

## 2. 文書構造の復元
### 段落・改行の最適化：
- 文の区切りを明確にする
- 段落の論理的な区切りを保持
- 不自然な改行を除去

### リスト・表構造の保持：
- 箇条書きの階層構造を維持
- 表のセル区切りを明確にする
- インデントの一貫性を確保

## 3. 言語品質の向上
### 自然な日本語への修正：
- 助詞の適切な使用（は/が、を/に など）
- 語尾の一致（である調、だ調の統一）
- 敬語表現の一貫性

### 読みやすさの向上：
- 長すぎる文の分割
- 冗長な表現の簡潔化
- 明確な文章構造

# 修正作業の手順

## Step 1: エラー検出
OCRテキストを注意深く読み、以下を特定：
- 明らかな誤認識文字
- 不自然な文章表現
- 構造的な問題（改行、段落など）

## Step 2: コンテキスト分析
- 文書全体の文脈を理解
- 専門分野・トピックを特定
- 想定される読者層を考慮

## Step 3: 段階的修正
1. 文字レベルの修正（誤認識文字）
2. 語句レベルの修正（語彙、表現）
3. 文レベルの修正（文法、構造）
4. 文書レベルの修正（段落、全体構成）

# 出力要件

**重要**: 修正されたテキストのみを出力してください。説明や注釈は不要です。

**修正対象テキスト**:
```
{ocr_text}
```

**修正結果**:"""

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
