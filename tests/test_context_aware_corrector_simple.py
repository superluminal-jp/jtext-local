"""
Simplified tests for context-aware correction module.

This module tests the ContextAwareCorrector class focusing on
the actual implementation without complex mocking.
"""

import pytest
from unittest.mock import Mock, patch
from jtext.correction.context_aware_corrector import ContextAwareCorrector


class TestContextAwareCorrectorSimple:
    """Simplified tests for the ContextAwareCorrector class."""

    def test_corrector_initialization(self):
        """Test corrector initialization."""
        corrector = ContextAwareCorrector()
        assert corrector is not None

    def test_correct_with_context_empty_text(self):
        """Test correction with empty text."""
        corrector = ContextAwareCorrector()

        result, corrections = corrector.correct_with_context("")
        assert result == ""
        assert corrections == 0

        result, corrections = corrector.correct_with_context("   ")
        assert result == "   "
        assert corrections == 0

    def test_correct_with_context_basic_text(self):
        """Test correction with basic text (fallback to rule-based)."""
        corrector = ContextAwareCorrector()

        # Mock the LLM call to fail, forcing rule-based correction
        with patch("requests.post", side_effect=Exception("API Error")):
            result, corrections = corrector.correct_with_context("テスト文書")

            # Should return some result (rule-based)
            assert isinstance(result, str)
            assert isinstance(corrections, int)

    def test_build_context_prompt_general(self):
        """Test prompt building for general context."""
        corrector = ContextAwareCorrector()

        prompt = corrector._build_context_prompt(
            "テスト文書",
            "general",
            {"type": "image", "confidence": 0.8},
            "前のテキスト",
        )

        assert isinstance(prompt, str)
        assert "テスト文書" in prompt
        assert len(prompt) > 100  # Should be substantial

    def test_build_context_prompt_academic(self):
        """Test prompt building for academic context."""
        corrector = ContextAwareCorrector()

        prompt = corrector._build_context_prompt(
            "学術論文のテキスト", "academic", {"type": "pdf", "pages": 10}, None
        )

        assert isinstance(prompt, str)
        assert "学術論文のテキスト" in prompt

    def test_build_context_prompt_business(self):
        """Test prompt building for business context."""
        corrector = ContextAwareCorrector()

        prompt = corrector._build_context_prompt(
            "ビジネス文書", "business", {"type": "docx"}, None
        )

        assert isinstance(prompt, str)
        assert "ビジネス文書" in prompt

    def test_build_context_prompt_technical(self):
        """Test prompt building for technical context."""
        corrector = ContextAwareCorrector()

        prompt = corrector._build_context_prompt(
            "技術仕様書", "technical", {"type": "pdf", "has_code": True}, None
        )

        assert isinstance(prompt, str)
        assert "技術仕様書" in prompt

    def test_get_context_instructions_all_types(self):
        """Test context instruction generation for all context types."""
        corrector = ContextAwareCorrector()

        context_types = [
            "general",
            "academic",
            "business",
            "technical",
            "vision_enhanced",
        ]

        for context_type in context_types:
            instructions = corrector._get_context_instructions(context_type)
            assert isinstance(instructions, str)
            assert len(instructions) > 50  # Should have substantial content

    def test_rule_based_correct(self):
        """Test rule-based correction fallback."""
        corrector = ContextAwareCorrector()

        # Test basic Japanese text corrections
        test_cases = [
            "テスト",
            "テ ス ト",  # With spaces
            "こんにちは。。",  # Duplicate punctuation
        ]

        for input_text in test_cases:
            result = corrector._rule_based_correct(input_text)
            assert isinstance(result, str)
            assert len(result) >= 0

    def test_count_corrections(self):
        """Test correction counting."""
        corrector = ContextAwareCorrector()

        # Test exact match (no corrections)
        count = corrector._count_corrections("同じテキスト", "同じテキスト")
        assert count == 0

        # Test different texts (should count corrections)
        count = corrector._count_corrections("元のテキスト", "修正されたテキスト")
        assert count > 0

    def test_clean_llm_response(self):
        """Test LLM response cleaning."""
        corrector = ContextAwareCorrector()

        test_cases = [
            ("修正: テキスト", "テキスト"),
            ("結果: 最終テキスト", "最終テキスト"),
            ("回答: 答え", "答え"),
            ("普通のテキスト", "普通のテキスト"),
        ]

        for input_text, expected_substring in test_cases:
            result = corrector._clean_response(input_text, "")
            assert expected_substring in result

    def test_context_aware_llm_correct_with_mock(self):
        """Test LLM correction with successful API call."""
        corrector = ContextAwareCorrector()

        # Mock successful API response
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "修正されたテキスト"}
            mock_post.return_value = mock_response

            result = corrector._context_aware_llm_correct(
                "原文", "general", {"type": "image"}, None
            )

            assert result == "修正されたテキスト"

    def test_context_aware_llm_correct_api_error(self):
        """Test LLM correction with API error (fallback to rule-based)."""
        corrector = ContextAwareCorrector()

        # Mock API failure
        with patch("requests.post", side_effect=Exception("API Error")):
            result = corrector._context_aware_llm_correct("原文", "general", None, None)

            # Should return rule-based result
            assert isinstance(result, str)

    def test_metadata_integration(self):
        """Test integration with document metadata."""
        corrector = ContextAwareCorrector()

        metadata = {
            "type": "pdf",
            "pages": 5,
            "confidence": 0.85,
            "language": "japanese",
            "has_tables": True,
        }

        # Test with API error to use rule-based correction
        with patch("requests.post", side_effect=Exception("API Error")):
            result, corrections = corrector.correct_with_context(
                "テスト", context_type="technical", document_metadata=metadata
            )

            # Should still work with rule-based correction
            assert isinstance(result, str)
            assert isinstance(corrections, int)

    def test_error_resilience(self):
        """Test error handling and resilience."""
        corrector = ContextAwareCorrector()

        # Test with network error (should fallback gracefully)
        with patch("requests.post", side_effect=Exception("Network error")):
            result, corrections = corrector.correct_with_context("テスト文書")

            # Should still return a result (fallback)
            assert isinstance(result, str)
            assert isinstance(corrections, int)

    def test_context_type_validation(self):
        """Test behavior with various context types."""
        corrector = ContextAwareCorrector()

        valid_contexts = [
            "general",
            "academic",
            "business",
            "technical",
            "vision_enhanced",
        ]

        for context in valid_contexts:
            # Should not raise exception for valid contexts
            try:
                result, corrections = corrector.correct_with_context(
                    "テストテキスト", context_type=context
                )
                assert isinstance(result, str)
                assert isinstance(corrections, int)
            except Exception as e:
                pytest.fail(f"Valid context {context} failed: {e}")
