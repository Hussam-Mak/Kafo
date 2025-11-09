"""
Main classification engine combining deterministic rules with Gemini AI reasoning.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.config_loader import config
from src.logger import logger
from src.document_processor import ProcessedDocument, PageContent
from src.pii_detector import PIIDetector, PIIDetectionResult, PIIType
from src.gemini_client import GeminiClient


class ClassificationCategory(Enum):
    """Document classification categories."""
    PUBLIC = "Public"
    CONFIDENTIAL = "Confidential"
    HIGHLY_SENSITIVE = "Highly Sensitive"
    UNSAFE = "Unsafe"


@dataclass
class CategoryScore:
    """Confidence score for a classification category."""
    category: ClassificationCategory
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    evidence_pages: List[int] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)  # "rule", "gemini", "pii"


@dataclass
class ClassificationResult:
    """Complete classification result for a document."""
    categories: List[ClassificationCategory] = field(default_factory=list)
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    primary_category: Optional[ClassificationCategory] = None
    overall_confidence: float = 0.0
    reasoning: str = ""
    pii_detections: Optional[PIIDetectionResult] = None
    page_citations: List[int] = field(default_factory=list)
    gemini_responses: List[Dict[str, Any]] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)


class DocumentClassifier:
    """Main document classifier combining rules and AI reasoning."""
    
    def __init__(self):
        """Initialize the document classifier."""
        self.pii_detector = PIIDetector()
        
        # Initialize Gemini client (may fail if API key not set)
        self.gemini_client = None
        try:
            self.gemini_client = GeminiClient()
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini client not available: {e}")
            logger.warning("Classification will use deterministic rules only")
        
        # Get classification settings
        self.categories = [
            ClassificationCategory(cat) 
            for cat in config.get('classification.categories', [
                "Public", "Confidential", "Highly Sensitive", "Unsafe"
            ])
        ]
        self.min_confidence = config.get('classification.min_confidence_threshold', 0.5)
    
    def _apply_deterministic_rules(
        self, 
        document: ProcessedDocument, 
        pii_result: PIIDetectionResult
    ) -> Dict[str, CategoryScore]:
        """Apply deterministic classification rules based on PII detection."""
        scores = {}
        
        # Initialize all categories with zero confidence
        for category in self.categories:
            scores[category.value] = CategoryScore(
                category=category,
                confidence=0.0,
                reasoning="",
                evidence_pages=[],
                evidence_sources=[]
            )
        
        # Rule 1: Highly Sensitive if PII detected
        if pii_result.total_detections > 0:
            high_sensitive_score = scores[ClassificationCategory.HIGHLY_SENSITIVE.value]
            
            # Calculate confidence based on number and type of PII
            pii_confidence = min(0.95, 0.5 + (pii_result.total_detections * 0.1))
            
            high_sensitive_score.confidence = max(
                high_sensitive_score.confidence,
                pii_confidence
            )
            high_sensitive_score.reasoning = (
                f"Detected {pii_result.total_detections} PII instances: "
                f"{', '.join(pii_result.pii_types_found)}"
            )
            high_sensitive_score.evidence_pages = pii_result.pages_with_pii
            high_sensitive_score.evidence_sources.append("pii")
            
            # Also mark pages with PII
            for page_num in pii_result.pages_with_pii:
                if page_num not in high_sensitive_score.evidence_pages:
                    high_sensitive_score.evidence_pages.append(page_num)
        
        # Rule 2: Check for specific high-sensitivity indicators
        sensitive_keywords = [
            "ssn", "social security", "credit card", "bank account",
            "passport", "driver's license", "medical record",
            "patient information", "hipaa", "gdpr"
        ]
        
        text_lower = document.full_text.lower()
        for keyword in sensitive_keywords:
            if keyword in text_lower:
                high_sensitive_score = scores[ClassificationCategory.HIGHLY_SENSITIVE.value]
                high_sensitive_score.confidence = max(
                    high_sensitive_score.confidence,
                    0.7
                )
                if "sensitive" not in high_sensitive_score.reasoning.lower():
                    high_sensitive_score.reasoning += f" Found sensitive keyword: {keyword}"
                high_sensitive_score.evidence_sources.append("rule")
        
        # Rule 3: Check for confidential indicators
        confidential_keywords = [
            "confidential", "internal use only", "proprietary",
            "not for distribution", "restricted", "classified"
        ]
        
        for keyword in confidential_keywords:
            if keyword in text_lower:
                conf_score = scores[ClassificationCategory.CONFIDENTIAL.value]
                conf_score.confidence = max(conf_score.confidence, 0.8)
                conf_score.reasoning = f"Found confidential indicator: {keyword}"
                conf_score.evidence_sources.append("rule")
        
        # Rule 4: Check for public indicators
        public_keywords = [
            "public", "marketing", "brochure", "advertisement",
            "press release", "newsletter", "public information"
        ]
        
        for keyword in public_keywords:
            if keyword in text_lower:
                pub_score = scores[ClassificationCategory.PUBLIC.value]
                pub_score.confidence = max(pub_score.confidence, 0.7)
                pub_score.reasoning = f"Found public indicator: {keyword}"
                pub_score.evidence_sources.append("rule")
        
        return scores
    
    def _classify_with_gemini(
        self, 
        document: ProcessedDocument
    ) -> Dict[str, Any]:
        """Classify document using Gemini AI."""
        if not self.gemini_client:
            return {}
        
        gemini_results = []
        
        # Process each page with Gemini
        for page in document.pages:
            if not page.text and not page.images:
                continue
            
            try:
                # Classify with text and images if available
                if page.images:
                    result = self.gemini_client.classify_with_images(
                        text=page.text,
                        images=page.images,
                        page_num=page.page_number,
                        total_pages=document.metadata.page_count
                    )
                else:
                    result = self.gemini_client.classify_text(
                        text=page.text,
                        page_num=page.page_number,
                        total_pages=document.metadata.page_count
                    )
                
                gemini_results.append(result)
                logger.debug(f"Gemini classification for page {page.page_number} completed")
            
            except Exception as e:
                error_msg = f"Error classifying page {page.page_number} with Gemini: {e}"
                logger.error(error_msg)
                gemini_results.append({
                    "error": error_msg,
                    "page": page.page_number
                })
        
        return {
            "gemini_results": gemini_results,
            "total_pages_processed": len(gemini_results)
        }
    
    def _combine_classification_results(
        self,
        rule_scores: Dict[str, CategoryScore],
        gemini_results: Dict[str, Any],
        pii_result: PIIDetectionResult
    ) -> ClassificationResult:
        """Combine deterministic rules and Gemini results."""
        final_scores = rule_scores.copy()
        
        # Process Gemini results
        if gemini_results and "gemini_results" in gemini_results:
            for gemini_result in gemini_results["gemini_results"]:
                if "error" in gemini_result:
                    continue
                
                page_num = gemini_result.get("page", 0)
                categories = gemini_result.get("categories", [])
                confidence_scores = gemini_result.get("confidence_scores", {})
                
                # Update scores based on Gemini results
                for category_name in categories:
                    if category_name in [cat.value for cat in self.categories]:
                        score = final_scores.get(category_name)
                        if score:
                            gemini_conf = confidence_scores.get(category_name, 0.7)
                            
                            # Combine confidences (weighted average)
                            if score.confidence > 0:
                                combined_conf = (score.confidence * 0.4) + (gemini_conf * 0.6)
                            else:
                                combined_conf = gemini_conf
                            
                            score.confidence = max(score.confidence, combined_conf)
                            
                            # Add Gemini reasoning
                            gemini_reasoning = gemini_result.get("reasoning", "")
                            if gemini_reasoning:
                                if score.reasoning:
                                    score.reasoning += f" | Gemini: {gemini_reasoning[:200]}"
                                else:
                                    score.reasoning = f"Gemini: {gemini_reasoning[:200]}"
                            
                            # Add evidence pages
                            if page_num not in score.evidence_pages:
                                score.evidence_pages.append(page_num)
                            
                            # Add evidence source
                            if "gemini" not in score.evidence_sources:
                                score.evidence_sources.append("gemini")
        
        # Determine final categories (above threshold)
        final_categories = []
        for category_name, score in final_scores.items():
            if score.confidence >= self.min_confidence:
                final_categories.append(score.category)
        
        # Determine primary category (highest confidence)
        primary_category = None
        max_confidence = 0.0
        for category_name, score in final_scores.items():
            if score.confidence > max_confidence:
                max_confidence = score.confidence
                primary_category = score.category
        
        # Calculate overall confidence
        if final_categories:
            overall_confidence = sum(
                final_scores[cat.value].confidence 
                for cat in final_categories
            ) / len(final_categories)
        else:
            overall_confidence = max_confidence if max_confidence > 0 else 0.0
        
        # Build reasoning
        reasoning_parts = []
        for category_name, score in final_scores.items():
            if score.confidence >= self.min_confidence:
                reasoning_parts.append(
                    f"{category_name} (confidence: {score.confidence:.2f}): {score.reasoning}"
                )
        
        combined_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No strong classification indicators found."
        
        # Collect all evidence pages
        all_evidence_pages = set()
        for score in final_scores.values():
            all_evidence_pages.update(score.evidence_pages)
        
        return ClassificationResult(
            categories=final_categories,
            category_scores=final_scores,
            primary_category=primary_category,
            overall_confidence=overall_confidence,
            reasoning=combined_reasoning,
            pii_detections=pii_result,
            page_citations=sorted(list(all_evidence_pages)),
            gemini_responses=gemini_results.get("gemini_results", []) if gemini_results else []
        )
    
    def classify(self, document: ProcessedDocument) -> ClassificationResult:
        """
        Classify a document using both deterministic rules and AI reasoning.
        
        Args:
            document: ProcessedDocument to classify
            
        Returns:
            ClassificationResult with categories, confidence scores, and reasoning
        """
        logger.info(f"Starting classification of {document.metadata.filename}")
        
        # Step 1: Detect PII
        pii_result = self.pii_detector.detect_pii(document)
        
        # Step 2: Apply deterministic rules
        rule_scores = self._apply_deterministic_rules(document, pii_result)
        
        # Step 3: Get Gemini classification (if available)
        gemini_results = {}
        if self.gemini_client:
            try:
                gemini_results = self._classify_with_gemini(document)
            except Exception as e:
                error_msg = f"Gemini classification failed: {e}"
                logger.error(error_msg)
                gemini_results = {"error": error_msg}
        
        # Step 4: Combine results
        result = self._combine_classification_results(
            rule_scores,
            gemini_results,
            pii_result
        )
        
        logger.info(
            f"Classification complete: {len(result.categories)} categories, "
            f"primary: {result.primary_category.value if result.primary_category else 'None'}, "
            f"confidence: {result.overall_confidence:.2f}"
        )
        
        return result

