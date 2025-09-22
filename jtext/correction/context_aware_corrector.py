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
        base_instructions = """あなたは高度なコンテキスト認識機能を持つテキスト修正専門家です。OCRまたは音声認識で抽出されたテキストを、文脈・文書タイプ・メタデータを総合的に分析し、最高品質の日本語テキストに修正してください。

# コンテキスト認識修正の原則

## 1. 多層的文脈分析
### マクロ文脈（文書レベル）:
- 文書の種類と目的の理解
- 想定読者層と専門性レベル
- 文書全体の論理構造と流れ

### ミクロ文脈（文・段落レベル）:
- 前後の文との関係性
- 段落内の論理的一貫性
- 語彙選択の適切性

### メタ文脈（外部情報）:
- 分野特有の専門用語・慣例
- 時代背景・地域性
- 文書フォーマットの標準

## 2. 階層的修正アプローチ
### Level 1: 基本的エラー修正
- 文字認識ミスの訂正
- 基本的な文法エラーの修正
- 明らかな誤字脱字の訂正

### Level 2: 文脈整合性の確保
- 文章の論理的一貫性の検証
- 専門用語の統一と正確性
- 文体・敬語レベルの統一

### Level 3: 品質最適化
- 読みやすさの向上
- 冗長性の除去
- 表現の自然性向上

## 3. 文書タイプ別最適化
### 技術文書:
- 正確性と明確性を最優先
- 専門用語の一貫した使用
- 手順・仕様の論理的配列

### 学術文書:
- 客観性と学術的表現の維持
- 引用・参考文献の適切な処理
- 論理的論証の構造保持

### ビジネス文書:
- 丁寧語・敬語の適切な使用
- 簡潔で効果的な表現
- 目的達成への焦点化

### 一般文書:
- 読みやすさと親しみやすさ
- 多様な読者層への配慮
- 自然な日本語表現

# 品質基準

## 必須達成項目:
✅ 文脈に完全に適合した修正
✅ 文書タイプに応じた適切な文体
✅ 専門用語・固有名詞の正確性
✅ 論理的一貫性の保持
✅ 自然で読みやすい日本語

## 禁止行為:
❌ 文脈を無視した機械的修正
❌ 元の意味・情報の改変
❌ 不適切な文体の混在
❌ 推測による情報追加
❌ 構造的整合性の破損

修正作業の基本方針を理解しました。以下のコンテキスト情報を基に、最適な修正を実行します。"""

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
# 🎓 学術文書専用修正戦略

## 言語品質基準:
- **専門用語**: 学術分野固有の用語を正確に使用
- **文体**: 客観的で論理的な「である調」を維持
- **構造**: 論理的論証の流れを重視

## 特別注意事項:
- 引用形式（APA、MLA等）の正確な保持
- 数値・統計データの精密性確保
- 仮説・結論の論理的整合性維持
- 参考文献リストの完全性

## 専門性の保持:
- 学術用語の一般化を避ける
- 専門概念の正確な表現
- 研究手法の適切な記述
""",
            "business": """
# 💼 ビジネス文書専用修正戦略

## 敬語・文体管理:
- **丁寧語**: 社内外に適した敬語レベル
- **簡潔性**: 要点を明確に伝える表現
- **一貫性**: 文書全体の敬語レベル統一

## ビジネス標準:
- 企業名・役職名の正確な表記
- 日付・時刻・数値の標準フォーマット
- 契約・提案書の法的表現の維持
- 会議録・報告書の構造的記述

## 効果的コミュニケーション:
- 読み手の立場を考慮した表現
- アクションアイテムの明確化
- 結論・提案の前面配置
""",
            "technical": """
# 🔧 技術文書専用修正戦略

## 技術精度の確保:
- **専門用語**: 技術分野固有の正確な用語使用
- **仕様記述**: 技術仕様の厳密な表現
- **手順説明**: 実行可能な明確な手順記述

## コード・コマンド保護:
- プログラムコードの文法的正確性
- システムコマンドの実行可能性
- API仕様・パラメータの正確性
- 設定ファイルの文法遵守

## 技術文書特有の構造:
- 前提条件・環境要件の明記
- エラーハンドリング・トラブルシューティング
- バージョン情報・互換性の記述
- 技術的制約・注意事項の強調
""",
            "general": """
# 📝 一般文書修正戦略

## 自然な日本語への最適化:
- **読みやすさ**: 平易で理解しやすい表現
- **流れ**: 自然な文章の流れと接続
- **語彙**: 適切な語彙レベルの選択

## 多様な読者への配慮:
- 専門用語の分かりやすい説明
- 文脈に応じた例示・比喩の活用
- 段落構成の論理性
- 結論・要点の明確化

## 表現の豊かさ:
- 単調な表現の回避
- 適切な修辞技法の使用
- 感情的配慮のある表現
- 文化的コンテキストの考慮
""",
            "vision_enhanced": """
# 👁️ Vision強化コンテキスト修正戦略

## マルチモーダル情報統合:
- **画像分析結果**: Vision分析の内容を最大限活用
- **構造整合性**: 画像レイアウトとテキスト構造の完全一致
- **視覚的コンテキスト**: 画像から読み取れる文脈情報の活用

## 高精度修正アプローチ:
- OCRエラーの画像情報による修正
- 文書タイプ特定情報の活用
- レイアウト構造の正確な再現
- 専門分野判定による用語修正

## Vision分析品質の活用:
- 文書の目的・読者層の推定活用
- 表・図表構造の詳細再現
- 文字品質情報による修正優先度決定
- 文書完全性の確保
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
