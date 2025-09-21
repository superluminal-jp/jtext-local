"""
Context-aware LLM correction for improved accuracy.

This module provides advanced correction capabilities that use context
information to improve the quality of OCR and ASR corrections.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import json
import requests

from ..utils.logging import get_logger


logger = get_logger(__name__)


class ContextAwareCorrector:
    """
    Context-aware LLM correction for improved accuracy.

    This class provides advanced correction capabilities that use context
    information to improve the quality of OCR and ASR corrections.
    """

    def __init__(self, model: str = "llama2", context_window: int = 2048):
        """
        Initialize the context-aware corrector.

        Args:
            model: LLM model name to use for correction
            context_window: Maximum context window size
        """
        self.model = model
        self.context_window = context_window
        self._model_available = self._check_model_availability()

        logger.info(f"Initialized ContextAwareCorrector with model: {model}")

    def correct_with_context(
        self,
        text: str,
        context_type: str = "general",
        document_metadata: Optional[Dict[str, Any]] = None,
        previous_text: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Correct text using context-aware LLM prompts.

        Args:
            text: Text to correct
            context_type: Type of context (general, academic, business, technical)
            document_metadata: Metadata about the source document
            previous_text: Previous text for context

        Returns:
            Tuple of (corrected_text, number_of_corrections)
        """
        if not text.strip():
            return text, 0

        logger.debug(f"Correcting text with context: {context_type}")

        if self._model_available:
            corrected_text = self._context_aware_llm_correct(
                text, context_type, document_metadata, previous_text
            )
        else:
            corrected_text = self._rule_based_correct(text)

        # Count corrections made
        corrections = self._count_corrections(text, corrected_text)

        logger.debug(f"Applied {corrections} corrections with context")
        return corrected_text, corrections

    def _context_aware_llm_correct(
        self,
        text: str,
        context_type: str,
        document_metadata: Optional[Dict[str, Any]],
        previous_text: Optional[str],
    ) -> str:
        """
        Correct text using context-aware LLM prompts.

        Args:
            text: Text to correct
            context_type: Type of context
            document_metadata: Document metadata
            previous_text: Previous text for context

        Returns:
            Context-aware corrected text
        """
        try:
            # Build context-aware prompt
            prompt = self._build_context_prompt(
                text, context_type, document_metadata, previous_text
            )

            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": self.context_window,
                },
            }

            logger.debug(f"Calling context-aware LLM with model: {self.model}")

            # Make request to Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60,  # Longer timeout for context-aware processing
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result.get("response", "").strip()

                # Clean up the response
                corrected_text = self._clean_response(corrected_text, prompt)

                logger.info(
                    f"Context-aware correction completed: {len(corrected_text)} characters"
                )
                return corrected_text
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return self._rule_based_correct(text)

        except Exception as e:
            logger.error(f"Context-aware correction failed: {e}")
            return self._rule_based_correct(text)

    def _build_context_prompt(
        self,
        text: str,
        context_type: str,
        document_metadata: Optional[Dict[str, Any]],
        previous_text: Optional[str],
    ) -> str:
        """
        Build context-aware prompt for LLM correction.

        Args:
            text: Text to correct
            context_type: Type of context
            document_metadata: Document metadata
            previous_text: Previous text for context

        Returns:
            Context-aware prompt
        """
        # Base correction instructions
        base_instructions = """
以下のテキストはOCRまたは音声認識で抽出されたものです。
文脈を考慮して、日本語として自然で正確な文章に修正してください。

修正方針：
1. 明らかな誤認識文字を修正
2. 文脈に合わない漢字を適切に修正
3. 句読点や改行を適切に配置
4. 表やリストの構造を復元
5. 元の意味を変えない範囲で修正
6. 文脈に応じた専門用語の修正
"""

        # Add context-specific instructions
        context_instructions = self._get_context_instructions(context_type)

        # Add document metadata context
        metadata_context = ""
        if document_metadata:
            metadata_context = self._format_metadata_context(document_metadata)

        # Add previous text context
        previous_context = ""
        if previous_text:
            previous_context = f"\n前の文脈:\n{previous_text[:500]}...\n"

        # Build final prompt
        prompt = f"""{base_instructions}

{context_instructions}

{metadata_context}

{previous_context}

元テキスト：
{text}

修正版："""

        return prompt

    def _get_context_instructions(self, context_type: str) -> str:
        """
        Get context-specific correction instructions.

        Args:
            context_type: Type of context

        Returns:
            Context-specific instructions
        """
        instructions = {
            "academic": """
学術文書の修正方針：
- 専門用語の正確性を重視
- 引用や参考文献の形式を保持
- 数式や記号の正確性を確保
- 学術的な文体を維持
""",
            "business": """
ビジネス文書の修正方針：
- 敬語や丁寧語の適切な使用
- ビジネス用語の正確性
- 日付や数値の正確性
- フォーマルな文体を維持
""",
            "technical": """
技術文書の修正方針：
- 技術用語の正確性を重視
- コードやコマンドの正確性
- 技術仕様の正確性
- 専門的な文体を維持
""",
            "general": """
一般的な文書の修正方針：
- 自然な日本語の表現
- 読みやすさを重視
- 文脈に応じた適切な表現
- 親しみやすい文体
""",
        }

        return instructions.get(context_type, instructions["general"])

    def _format_metadata_context(self, metadata: Dict[str, Any]) -> str:
        """
        Format document metadata for context.

        Args:
            metadata: Document metadata

        Returns:
            Formatted metadata context
        """
        context_parts = []

        if "format" in metadata:
            context_parts.append(f"文書形式: {metadata['format']}")

        if "pages" in metadata:
            context_parts.append(f"ページ数: {metadata['pages']}")

        if "has_tables" in metadata and metadata["has_tables"]:
            context_parts.append("表が含まれています")

        if "has_images" in metadata and metadata["has_images"]:
            context_parts.append("画像が含まれています")

        if "language" in metadata:
            context_parts.append(f"言語: {metadata['language']}")

        if context_parts:
            return f"文書情報:\n" + "\n".join(context_parts) + "\n"

        return ""

    def _clean_response(self, response: str, prompt: str) -> str:
        """
        Clean up LLM response.

        Args:
            response: Raw LLM response
            prompt: Original prompt

        Returns:
            Cleaned response
        """
        # Remove prompt if it was included in response
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "修正版：",
            "修正されたテキスト：",
            "修正結果：",
            "Corrected text:",
            "修正:",
        ]

        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()

        return response

    def _check_model_availability(self) -> bool:
        """
        Check if the specified LLM model is available.

        Returns:
            True if model is available, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]

                if self.model in available_models:
                    logger.info(f"Context-aware model {self.model} is available")
                    return True
                else:
                    logger.warning(
                        f"Model {self.model} not found. Available: {available_models}"
                    )
                    return False
            else:
                logger.warning("Ollama API not responding")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def _rule_based_correct(self, text: str) -> str:
        """
        Apply rule-based corrections as fallback.

        Args:
            text: Text to correct

        Returns:
            Rule-based corrected text
        """
        # Basic rule-based corrections
        corrected = text

        # Common OCR error corrections
        corrections = {
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
            "，": "、",
            "．": "。",
            "：": "：",
            "；": "；",
            "　　": "　",
            "  ": " ",
        }

        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)

        return corrected

    def _count_corrections(self, original: str, corrected: str) -> int:
        """
        Count the number of corrections made.

        Args:
            original: Original text
            corrected: Corrected text

        Returns:
            Number of corrections made
        """
        if len(original) != len(corrected):
            return abs(len(original) - len(corrected))

        return sum(1 for a, b in zip(original, corrected) if a != b)
