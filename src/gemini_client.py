"""
Gemini 1.5 Flash API client for document classification.
"""

import json
import base64
from typing import List, Dict, Any, Optional
from io import BytesIO
from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from src.config_loader import config
from src.logger import logger
from config.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
    IMAGE_ANALYSIS_PROMPT
)


class GeminiClient:
    """Client for interacting with Gemini 1.5 Flash API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        # Get API key from config
        api_key = config.get_gemini_api_key()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Set it in .env file or environment."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Get model configuration
        self.model_name = config.get('gemini.model', 'gemini-1.5-flash')
        self.temperature = config.get('gemini.temperature', 0.3)
        self.max_tokens = config.get('gemini.max_tokens', 2048)
        
        # Get safety settings
        safety_settings_config = config.get('gemini.safety_settings', {})
        self.safety_settings = self._parse_safety_settings(safety_settings_config)
        
        # Initialize model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings
            )
            logger.info(f"Gemini model '{self.model_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def _parse_safety_settings(self, settings: Dict[str, str]) -> List[Dict[str, Any]]:
        """Parse safety settings from config."""
        safety_map = {
            'HARM_CATEGORY_HARASSMENT': genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            'HARM_CATEGORY_HATE_SPEECH': genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            'HARM_CATEGORY_DANGEROUS_CONTENT': genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        }
        
        threshold_map = {
            'BLOCK_NONE': genai.types.HarmBlockThreshold.BLOCK_NONE,
            'BLOCK_ONLY_HIGH': genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            'BLOCK_MEDIUM_AND_ABOVE': genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            'BLOCK_LOW_AND_ABOVE': genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        
        safety_settings = []
        for category_name, threshold_name in settings.items():
            category = safety_map.get(category_name)
            threshold = threshold_map.get(threshold_name)
            
            if category and threshold:
                safety_settings.append({
                    "category": category,
                    "threshold": threshold
                })
        
        return safety_settings if safety_settings else None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for Gemini API."""
        buffered = BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def _prepare_image_for_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """Prepare image for Gemini API."""
        # Gemini API accepts PIL Images directly
        return image
    
    def classify_text(self, text: str, page_num: int, total_pages: int) -> Dict[str, Any]:
        """
        Classify text content using Gemini.
        
        Args:
            text: Text content to classify
            page_num: Current page number
            total_pages: Total number of pages
            
        Returns:
            Classification result dictionary
        """
        try:
            # Build prompt
            prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
                page_count=total_pages,
                doc_type="PDF",
                page_num=page_num,
                text_content=text[:8000],  # Limit text length
                image_context=""
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            # Parse response
            result = self._parse_classification_response(response.text, page_num)
            return result
            
        except Exception as e:
            logger.error(f"Error classifying text on page {page_num}: {e}")
            return {
                "error": str(e),
                "page": page_num,
                "categories": [],
                "confidence": 0.0
            }
    
    def classify_with_images(
        self, 
        text: str, 
        images: List[Image.Image], 
        page_num: int, 
        total_pages: int
    ) -> Dict[str, Any]:
        """
        Classify document with both text and images.
        
        Args:
            text: Text content
            images: List of PIL Images
            page_num: Current page number
            total_pages: Total number of pages
            
        Returns:
            Classification result dictionary
        """
        try:
            # Prepare content parts
            parts = []
            
            # Add text prompt
            prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
                page_count=total_pages,
                doc_type="PDF",
                page_num=page_num,
                text_content=text[:4000] if text else "No text content available.",
                image_context=f"\n\n{IMAGE_ANALYSIS_PROMPT}\n\nAnalyzing {len(images)} image(s) from this page."
            )
            parts.append(prompt)
            
            # Add images
            for img in images:
                prepared_img = self._prepare_image_for_gemini(img)
                parts.append(prepared_img)
            
            # Generate response
            response = self.model.generate_content(
                parts,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            # Parse response
            result = self._parse_classification_response(response.text, page_num)
            return result
            
        except Exception as e:
            logger.error(f"Error classifying with images on page {page_num}: {e}")
            return {
                "error": str(e),
                "page": page_num,
                "categories": [],
                "confidence": 0.0
            }
    
    def _parse_classification_response(self, response_text: str, page_num: int) -> Dict[str, Any]:
        """
        Parse Gemini's classification response.
        
        Args:
            response_text: Raw response text from Gemini
            page_num: Page number for context
            
        Returns:
            Parsed classification result
        """
        result = {
            "page": page_num,
            "categories": [],
            "confidence_scores": {},
            "reasoning": "",
            "citations": [],
            "raw_response": response_text
        }
        
        try:
            # Try to extract JSON from response
            # Look for JSON block in markdown code fences
            json_match = None
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end > json_start:
                    json_match = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                if json_end > json_start:
                    json_match = response_text[json_start:json_end].strip()
            
            # Try to parse as JSON
            if json_match:
                try:
                    parsed = json.loads(json_match)
                    result.update({
                        "categories": parsed.get("categories", []),
                        "confidence_scores": parsed.get("confidence_scores", {}),
                        "reasoning": parsed.get("reasoning", ""),
                        "citations": parsed.get("citations", [])
                    })
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from response, using text parsing")
                    result = self._parse_text_response(response_text, page_num)
            else:
                # Fallback to text parsing
                result = self._parse_text_response(response_text, page_num)
        
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            result["reasoning"] = response_text
            result["error"] = str(e)
        
        return result
    
    def _parse_text_response(self, response_text: str, page_num: int) -> Dict[str, Any]:
        """Parse text-based response when JSON parsing fails."""
        result = {
            "page": page_num,
            "categories": [],
            "confidence_scores": {},
            "reasoning": response_text,
            "citations": [page_num]
        }
        
        # Try to extract categories from text
        categories = ["Public", "Confidential", "Highly Sensitive", "Unsafe"]
        for category in categories:
            if category.lower() in response_text.lower():
                result["categories"].append(category)
                # Try to extract confidence if mentioned
                confidence_match = None
                if f"{category.lower()}" in response_text.lower():
                    # Look for confidence scores near the category
                    import re
                    conf_pattern = rf"{re.escape(category)}.*?(\d+\.?\d*)"
                    match = re.search(conf_pattern, response_text, re.IGNORECASE)
                    if match:
                        try:
                            conf = float(match.group(1))
                            if 0 <= conf <= 1:
                                result["confidence_scores"][category] = conf
                            elif 0 <= conf <= 100:
                                result["confidence_scores"][category] = conf / 100
                        except:
                            pass
                
                # Default confidence if not found
                if category not in result["confidence_scores"]:
                    result["confidence_scores"][category] = 0.7
        
        # If no categories found, default to analyzing the reasoning
        if not result["categories"]:
            # Try to infer from reasoning text
            text_lower = response_text.lower()
            if any(word in text_lower for word in ["public", "marketing", "brochure", "advertisement"]):
                result["categories"].append("Public")
                result["confidence_scores"]["Public"] = 0.6
            elif any(word in text_lower for word in ["confidential", "internal", "private", "proprietary"]):
                result["categories"].append("Confidential")
                result["confidence_scores"]["Confidential"] = 0.6
            elif any(word in text_lower for word in ["pii", "sensitive", "personal", "ssn", "credit card"]):
                result["categories"].append("Highly Sensitive")
                result["confidence_scores"]["Highly Sensitive"] = 0.6
            elif any(word in text_lower for word in ["unsafe", "explicit", "violent", "inappropriate"]):
                result["categories"].append("Unsafe")
                result["confidence_scores"]["Unsafe"] = 0.6
        
        return result

