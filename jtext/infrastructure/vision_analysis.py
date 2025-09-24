"""
LLM-based Vision Analysis Service.

Implements comprehensive vision analysis using LLM models for document understanding,
layout analysis, and content extraction. Supports both Ollama integration for local
LLM inference and multimodal analysis.

Features:
- Document structure analysis
- Text region detection and classification
- Visual element identification
- Quality assessment and enhancement suggestions
- Content categorization
- Multimodal LLM integration

@author: jtext Development Team
@since: 1.0.0
@compliance: AGENTS.md, Clean Architecture, DDD
"""

import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from ..core import Result, Ok, Err
from ..domain import (
    Language,
    Confidence,
    VisionAnalysis,
    VisionAnalysisId,
    VisionAnalysisType,
    DocumentId,
    TextRegion,
    VisualElement,
    QualityAssessment,
    DocumentStructure,
)
from .logging import get_logger, start_performance_tracking

# Constants
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 5
EXTENDED_TIMEOUT = 300
DEFAULT_CONFIDENCE = 0.5
DEFAULT_ANALYSIS_CONFIDENCE = 0.8
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40


@dataclass
class VisionAnalysisResult:
    """Result of vision analysis."""

    analysis: VisionAnalysis
    processing_time_ms: float
    model_used: str
    confidence: Confidence


class OllamaVisionAnalysisService:
    """
    Ollama-based vision analysis service.

    Implements comprehensive document analysis using local Ollama deployment
    with multimodal capabilities for document understanding and analysis.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        """Initialize Ollama vision analysis service."""
        self.base_url = base_url
        self.logger = get_logger("jtext.infrastructure.vision_analysis")

        # Test Ollama connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            if not HAS_REQUESTS:
                self.logger.warn(
                    "Requests library not available, vision analysis disabled",
                    operation="connection_test",
                    component="ollama",
                    tags={"category": "system", "result": "dependency_missing"},
                )
                return False

            response = requests.get(
                f"{self.base_url}/api/tags", timeout=DEFAULT_TIMEOUT
            )

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

    def analyze_document(
        self,
        image_path: str,
        language: Language,
        analysis_type: str = "comprehensive",
        model_name: Optional[str] = None,
    ) -> Result[VisionAnalysis, str]:
        """
        Analyze document using LLM-based vision analysis.

        Performs comprehensive document analysis including structure detection,
        text region identification, visual element recognition, and quality assessment.

        Args:
            image_path: Path to image file
            language: Document language
            analysis_type: Type of analysis to perform
            model_name: LLM model to use

        Returns:
            Result containing vision analysis or error message
        """
        metrics = start_performance_tracking()

        try:
            self.logger.info(
                "Starting vision analysis",
                operation="vision_analysis",
                component="ollama",
                image_path=image_path,
                language=language.value,
                analysis_type=analysis_type,
                model=model_name or "default",
                tags={"category": "processing", "operation": "vision_analysis"},
            )

            # Generate comprehensive analysis prompt
            prompt = self._generate_analysis_prompt(language, analysis_type)

            # Select appropriate model
            model = self._select_model(model_name, language)

            # Call Ollama multimodal API
            result = self._call_ollama_multimodal_api(model, prompt, image_path)

            if result.is_err:
                return result

            # Parse and validate result
            parsed_result = self._parse_analysis_response(
                result.unwrap(), language, analysis_type, model, metrics
            )

            if parsed_result.is_ok:
                analysis = parsed_result.unwrap()

                self.logger.info(
                    "Vision analysis completed successfully",
                    operation="vision_analysis",
                    component="ollama",
                    analysis_type=analysis.analysis_type.value,
                    confidence=analysis.confidence.to_float(),
                    processing_time_ms=analysis.processing_time_ms,
                    tags={"category": "processing", "result": "success"},
                )

                return Ok(analysis)
            else:
                return parsed_result

        except Exception as e:
            metrics.finish()
            error_msg = f"Vision analysis failed: {str(e)}"

            self.logger.error(
                "Vision analysis failed with exception",
                operation="vision_analysis",
                component="ollama",
                error=str(e),
                processing_time_ms=metrics.duration_ms or 0.0,
                tags={"category": "system_error", "result": "failure"},
            )

            return Err(error_msg)

    def _generate_analysis_prompt(self, language: Language, analysis_type: str) -> str:
        """Generate comprehensive analysis prompt."""
        language_context = self._get_language_context(language)
        requirements = self._get_analysis_requirements()
        json_format = self._get_json_format_template()

        return f"""You are an expert document analysis specialist. Your task is to perform comprehensive vision analysis of the provided document image.

{language_context}

{requirements}

{json_format}

Focus on accuracy and provide detailed, actionable insights about the document."""

    def _get_language_context(self, language: Language) -> str:
        """Get language-specific context for analysis."""
        if language == Language.JAPANESE:
            return """
Japanese Document Analysis Context:
- Document types: Business letters, forms, academic papers, manga, newspapers
- Layout patterns: Vertical text (tategaki), horizontal text (yokogaki), mixed layouts
- Common elements: Headers, body text, captions, tables, charts, diagrams
- Text regions: Titles, paragraphs, captions, annotations, footnotes
- Visual elements: Images, charts, tables, diagrams, logos, stamps
"""
        else:
            return """
English Document Analysis Context:
- Document types: Business documents, academic papers, forms, reports, presentations
- Layout patterns: Single column, multi-column, mixed layouts
- Common elements: Headers, body text, captions, tables, charts, diagrams
- Text regions: Titles, paragraphs, captions, annotations, footnotes
- Visual elements: Images, charts, tables, diagrams, logos, signatures
"""

    def _get_analysis_requirements(self) -> str:
        """Get analysis requirements section."""
        return """ANALYSIS REQUIREMENTS:
1. Document Structure Analysis:
   - Identify document type (letter, form, report, academic paper, etc.)
   - Detect main sections and layout structure
   - Analyze reading direction and text flow
   - Identify headers, footers, and margins

2. Text Region Detection:
   - Identify all text regions in the document
   - Classify each region by content type (title, body, caption, etc.)
   - Estimate confidence for each text region
   - Note any special formatting or emphasis

3. Visual Element Recognition:
   - Identify images, charts, tables, diagrams
   - Detect logos, stamps, signatures
   - Recognize decorative elements
   - Note positioning and relationships

4. Quality Assessment:
   - Evaluate image clarity and contrast
   - Assess text readability
   - Identify potential enhancement opportunities
   - Note any quality issues

5. Content Categorization:
   - Determine primary content categories
   - Identify document purpose and context
   - Note any special characteristics"""

    def _get_json_format_template(self) -> str:
        """Get JSON format template for analysis response."""
        return """Please provide your analysis in this exact JSON format:
{{
    "document_structure": {{
        "document_type": "business_letter",
        "main_sections": ["header", "body", "signature"],
        "layout_type": "single_column",
        "reading_direction": "left-to-right"
    }},
    "text_regions": [
        {{
            "area": "header",
            "content_type": "title",
            "confidence": 0.95,
            "bounding_box": {{"x": 0, "y": 0, "width": 100, "height": 20}},
            "text_content": "Document title or header text"
        }},
        {{
            "area": "body",
            "content_type": "paragraph",
            "confidence": 0.88,
            "bounding_box": {{"x": 0, "y": 20, "width": 100, "height": 60}},
            "text_content": "Main body text content"
        }}
    ],
    "visual_elements": [
        {{
            "element_type": "image",
            "description": "Company logo or diagram",
            "confidence": 0.92,
            "position": {{"x": 10, "y": 10}},
            "size": {{"width": 30, "height": 20}}
        }}
    ],
    "quality_assessment": {{
        "clarity": 0.85,
        "contrast": 0.90,
        "readability": 0.88,
        "overall_quality": 0.87,
        "enhancement_suggestions": ["Improve contrast", "Enhance text sharpness"]
    }},
    "content_categories": ["business", "formal", "correspondence"],
    "analysis_confidence": 0.89,
    "reasoning": "Comprehensive analysis of document structure, content, and quality"
}}"""

    def _select_model(self, requested_model: Optional[str], language: Language) -> str:
        """Select appropriate model for vision analysis."""
        if requested_model:
            return requested_model

        # Default model selection based on language
        if language == Language.JAPANESE:
            return "gemma3:4b"  # Good for Japanese text understanding
        else:
            return "gemma3:4b"  # Good general-purpose model

    def _call_ollama_multimodal_api(
        self, model_name: str, prompt: str, image_path: str
    ) -> Result[str, str]:
        """Call Ollama API for multimodal vision analysis."""
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
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "top_k": DEFAULT_TOP_K,
                },
            }

            self.logger.debug(
                "Calling Ollama API for vision analysis",
                operation="ollama_multimodal_api",
                component="ollama",
                model=model_name,
                prompt_length=len(prompt),
                image_path=image_path,
                tags={"category": "api", "operation": "vision_analysis"},
            )

            # Make API call with extended timeout for multimodal processing
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=api_request,
                timeout=EXTENDED_TIMEOUT,  # Extended timeout for multimodal processing
            )

            if response.status_code == 200:
                result = response.json()
                return Ok(result.get("response", ""))
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                self.logger.error(
                    "Ollama vision analysis API call failed",
                    operation="ollama_multimodal_api",
                    component="ollama",
                    status_code=response.status_code,
                    error=response.text,
                    tags={"category": "api_error"},
                )
                return Err(error_msg)

        except requests.exceptions.Timeout:
            return Err(
                f"Ollama API timeout after {EXTENDED_TIMEOUT}s - vision analysis took too long"
            )
        except requests.exceptions.ConnectionError:
            return Err(
                f"Ollama API connection failed to {self.base_url} - check if Ollama is running"
            )
        except requests.exceptions.RequestException as e:
            return Err(f"Ollama API request failed: {str(e)}")
        except Exception as e:
            return Err(f"Unexpected error calling Ollama vision analysis API: {str(e)}")

    def _encode_image_to_base64(self, image_path: str) -> Result[str, str]:
        """Encode image to base64 for multimodal API."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode("utf-8")
                return Ok(base64_encoded)

        except FileNotFoundError:
            return Err(f"Image file not found: {image_path}")
        except Exception as e:
            return Err(f"Failed to encode image: {str(e)}")

    def _parse_analysis_response(
        self,
        response: str,
        language: Language,
        analysis_type: str,
        model_used: str,
        metrics,
    ) -> Result[VisionAnalysis, str]:
        """Parse and validate vision analysis response."""
        try:
            metrics.finish()

            # Try to extract JSON from response
            response = response.strip()

            # Find JSON block if wrapped in other text
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                return Err("No valid JSON found in LLM response")

            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)

            # Create VisionAnalysis entity
            analysis_id = VisionAnalysisId.generate()
            document_id = DocumentId.generate()

            # Parse document structure
            layout_structure = None
            if "document_structure" in parsed:
                structure_data = parsed["document_structure"]
                layout_structure = DocumentStructure(
                    document_type=structure_data.get("document_type", "unknown"),
                    main_sections=structure_data.get("main_sections", []),
                    layout_type=structure_data.get("layout_type", "unknown"),
                    reading_direction=structure_data.get(
                        "reading_direction", "left-to-right"
                    ),
                )

            # Parse text regions
            text_regions = []
            if "text_regions" in parsed:
                for region_data in parsed["text_regions"]:
                    region = TextRegion(
                        area=region_data.get("area", "unknown"),
                        content_type=region_data.get("content_type", "unknown"),
                        confidence=Confidence.from_float(
                            region_data.get("confidence", DEFAULT_CONFIDENCE)
                        ),
                        bounding_box=region_data.get("bounding_box", {}),
                        text_content=region_data.get("text_content", ""),
                    )
                    text_regions.append(region)

            # Parse visual elements
            visual_elements = []
            if "visual_elements" in parsed:
                for element_data in parsed["visual_elements"]:
                    element = VisualElement(
                        element_type=element_data.get("element_type", "unknown"),
                        description=element_data.get("description", ""),
                        confidence=Confidence.from_float(
                            element_data.get("confidence", DEFAULT_CONFIDENCE)
                        ),
                        position=element_data.get("position", {}),
                        size=element_data.get("size", {}),
                    )
                    visual_elements.append(element)

            # Parse quality assessment
            quality_assessment = None
            if "quality_assessment" in parsed:
                quality_data = parsed["quality_assessment"]
                quality_assessment = QualityAssessment(
                    clarity=quality_data.get("clarity", DEFAULT_CONFIDENCE),
                    contrast=quality_data.get("contrast", DEFAULT_CONFIDENCE),
                    readability=quality_data.get("readability", DEFAULT_CONFIDENCE),
                    overall_quality=quality_data.get(
                        "overall_quality", DEFAULT_CONFIDENCE
                    ),
                    enhancement_suggestions=quality_data.get(
                        "enhancement_suggestions", []
                    ),
                )

            # Parse content categories
            content_categories = parsed.get("content_categories", [])

            # Create analysis confidence
            analysis_confidence = Confidence.from_float(
                parsed.get("analysis_confidence", DEFAULT_ANALYSIS_CONFIDENCE)
            )

            # Create VisionAnalysis entity
            analysis = VisionAnalysis(
                id=analysis_id,
                document_id=document_id,
                analysis_type=VisionAnalysisType.COMPREHENSIVE,
                confidence=analysis_confidence,
                processing_time_ms=metrics.duration_ms or 0.0,
                model_used=model_used,
                created_at=datetime.now(timezone.utc),
                layout_structure=layout_structure,
                text_regions=text_regions,
                visual_elements=visual_elements,
                quality_assessment=quality_assessment,
                content_categories=content_categories,
                metadata={
                    "reasoning": parsed.get("reasoning", "Vision analysis completed"),
                    "language": language.value,
                    "analysis_type": analysis_type,
                },
            )

            return Ok(analysis)

        except json.JSONDecodeError as e:
            return Err(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            return Err(f"Failed to create vision analysis: {str(e)}")

    def is_available(self) -> bool:
        """Check if vision analysis service is available."""
        return HAS_REQUESTS and self._test_connection()


def get_vision_analysis_service(
    base_url: str = DEFAULT_BASE_URL,
) -> OllamaVisionAnalysisService:
    """Get vision analysis service instance.

    Args:
        base_url: Ollama server base URL

    Returns:
        Configured vision analysis service instance
    """
    return OllamaVisionAnalysisService(base_url)
