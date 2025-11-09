"""
Output formatter for classification results.
Creates structured JSON output with classifications, confidence scores, citations, and reasoning.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from src.config_loader import config
from src.logger import logger
from src.classifier import ClassificationResult, CategoryScore, ClassificationCategory
from src.document_processor import ProcessedDocument
from src.pii_detector import PIIDetectionResult, PIIDetection


class OutputFormatter:
    """Formats classification results into structured JSON output."""
    
    def __init__(self):
        """Initialize the output formatter."""
        self.output_dir = Path(config.get('paths.output_dir', './data/output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_classification_result(
        self,
        classification_result: ClassificationResult,
        document: ProcessedDocument
    ) -> Dict[str, Any]:
        """
        Format classification result into structured JSON.
        
        Args:
            classification_result: ClassificationResult object
            document: ProcessedDocument object
            
        Returns:
            Dictionary with structured classification output
        """
        # Build category scores
        category_scores = {}
        for category_name, score in classification_result.category_scores.items():
            if score.confidence > 0:
                category_scores[category_name] = {
                    "confidence": round(score.confidence, 3),
                    "reasoning": score.reasoning,
                    "evidence_pages": sorted(score.evidence_pages),
                    "evidence_sources": score.evidence_sources
                }
        
        # Build PII detections summary
        pii_summary = None
        if classification_result.pii_detections:
            pii_summary = self._format_pii_detections(
                classification_result.pii_detections
            )
        
        # Build page citations with evidence
        page_citations = self._build_page_citations(
            classification_result,
            document
        )
        
        # Build output structure
        output = {
            "document_metadata": {
                "filename": document.metadata.filename,
                "file_path": str(document.metadata.file_path),
                "file_size": document.metadata.file_size,
                "page_count": document.metadata.page_count,
                "processing_timestamp": datetime.now().isoformat()
            },
            "classification": {
                "categories": [cat.value for cat in classification_result.categories],
                "primary_category": (
                    classification_result.primary_category.value 
                    if classification_result.primary_category 
                    else None
                ),
                "overall_confidence": round(classification_result.overall_confidence, 3),
                "category_scores": category_scores
            },
            "reasoning": {
                "summary": classification_result.reasoning,
                "model_reasoning": self._extract_gemini_reasoning(
                    classification_result.gemini_responses
                )
            },
            "evidence": {
                "page_citations": page_citations,
                "total_evidence_pages": len(classification_result.page_citations),
                "evidence_sources": self._get_evidence_sources(classification_result)
            },
            "pii_detections": pii_summary,
            "processing_info": {
                "processing_errors": classification_result.processing_errors,
                "gemini_enabled": len(classification_result.gemini_responses) > 0,
                "total_pages_analyzed": document.metadata.page_count
            }
        }
        
        return output
    
    def _format_pii_detections(
        self, 
        pii_result: PIIDetectionResult
    ) -> Dict[str, Any]:
        """Format PII detection results."""
        # Group detections by type
        detections_by_type = {}
        for detection in pii_result.detections:
            pii_type = detection.pii_type.value
            if pii_type not in detections_by_type:
                detections_by_type[pii_type] = []
            
            detections_by_type[pii_type].append({
                "value": detection.value,  # Already masked
                "confidence": round(detection.confidence, 3),
                "page": detection.page_number,
                "position": {
                    "start": detection.start_position,
                    "end": detection.end_position
                },
                "context": detection.context[:200] if detection.context else ""  # Limit context length
            })
        
        return {
            "total_detections": pii_result.total_detections,
            "pii_types_found": pii_result.pii_types_found,
            "pages_with_pii": sorted(pii_result.pages_with_pii),
            "average_confidence": round(pii_result.confidence_score, 3),
            "detections_by_type": detections_by_type
        }
    
    def _build_page_citations(
        self,
        classification_result: ClassificationResult,
        document: ProcessedDocument
    ) -> List[Dict[str, Any]]:
        """Build detailed page citations with evidence."""
        citations = []
        
        # Get all evidence pages
        evidence_pages = set(classification_result.page_citations)
        
        # Add PII evidence pages
        if classification_result.pii_detections:
            evidence_pages.update(classification_result.pii_detections.pages_with_pii)
        
        # Build citation for each page
        for page_num in sorted(evidence_pages):
            page = None
            for p in document.pages:
                if p.page_number == page_num:
                    page = p
                    break
            
            if not page:
                continue
            
            # Get evidence for this page
            page_evidence = {
                "page_number": page_num,
                "has_text": page.has_text,
                "is_scanned": page.is_scanned,
                "evidence_sources": [],
                "evidence_summary": []
            }
            
            # Add category evidence
            for category_name, score in classification_result.category_scores.items():
                if page_num in score.evidence_pages:
                    page_evidence["evidence_sources"].extend(score.evidence_sources)
                    page_evidence["evidence_summary"].append(
                        f"{category_name} (confidence: {score.confidence:.2f})"
                    )
            
            # Add PII evidence
            if classification_result.pii_detections:
                page_pii = [
                    d for d in classification_result.pii_detections.detections
                    if d.page_number == page_num
                ]
                if page_pii:
                    page_evidence["evidence_sources"].append("pii")
                    page_evidence["evidence_summary"].append(
                        f"PII detected: {len(page_pii)} instance(s)"
                    )
            
            # Remove duplicates
            page_evidence["evidence_sources"] = list(set(page_evidence["evidence_sources"]))
            
            # Add text preview
            if page.text:
                page_evidence["text_preview"] = page.text[:300]  # First 300 chars
            
            citations.append(page_evidence)
        
        return citations
    
    def _extract_gemini_reasoning(
        self, 
        gemini_responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract reasoning from Gemini responses."""
        reasoning_list = []
        
        for response in gemini_responses:
            if "error" in response:
                continue
            
            reasoning_list.append({
                "page": response.get("page", 0),
                "reasoning": response.get("reasoning", ""),
                "categories": response.get("categories", []),
                "confidence_scores": response.get("confidence_scores", {})
            })
        
        return reasoning_list
    
    def _get_evidence_sources(
        self, 
        classification_result: ClassificationResult
    ) -> List[str]:
        """Get all unique evidence sources."""
        sources = set()
        
        # From category scores
        for score in classification_result.category_scores.values():
            sources.update(score.evidence_sources)
        
        # From PII detections
        if classification_result.pii_detections:
            sources.add("pii")
        
        # From Gemini
        if classification_result.gemini_responses:
            sources.add("gemini")
        
        return sorted(list(sources))
    
    def save_json_output(
        self,
        output: Dict[str, Any],
        filename: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save classification output to JSON file.
        
        Args:
            output: Formatted output dictionary
            filename: Optional filename (defaults to document filename + timestamp)
            output_dir: Optional output directory (defaults to configured output dir)
            
        Returns:
            Path to saved JSON file
        """
        if not filename:
            doc_filename = output["document_metadata"]["filename"]
            base_name = Path(doc_filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_classification_{timestamp}.json"
        
        # Use provided output_dir or default
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = save_dir / filename
        
        # Save with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Classification output saved to: {output_path}")
        
        return output_path
    
    def format_for_api(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format output for API response (simplified version).
        
        Args:
            output: Full output dictionary
            
        Returns:
            Simplified API response dictionary
        """
        return {
            "status": "success",
            "classification": output["classification"],
            "reasoning": output["reasoning"]["summary"],
            "evidence": {
                "page_citations": [c["page_number"] for c in output["evidence"]["page_citations"]],
                "total_pages": output["evidence"]["total_evidence_pages"]
            },
            "pii_detections": {
                "total": output["pii_detections"]["total_detections"] if output["pii_detections"] else 0,
                "types": output["pii_detections"]["pii_types_found"] if output["pii_detections"] else []
            }
        }
    
    def create_human_readable_summary(
        self,
        output: Dict[str, Any]
    ) -> str:
        """
        Create a human-readable summary of classification results.
        
        Args:
            output: Formatted output dictionary
            
        Returns:
            Human-readable summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DOCUMENT CLASSIFICATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Document info
        meta = output["document_metadata"]
        lines.append(f"Document: {meta['filename']}")
        lines.append(f"Pages: {meta['page_count']}")
        lines.append(f"Processed: {meta['processing_timestamp']}")
        lines.append("")
        
        # Classification
        classification = output["classification"]
        lines.append("CLASSIFICATION RESULTS")
        lines.append("-" * 60)
        lines.append(f"Primary Category: {classification['primary_category']}")
        lines.append(f"Overall Confidence: {classification['overall_confidence']:.2%}")
        lines.append(f"Categories: {', '.join(classification['categories'])}")
        lines.append("")
        
        # Category scores
        if classification['category_scores']:
            lines.append("Category Scores:")
            for category, details in classification['category_scores'].items():
                lines.append(f"  {category}: {details['confidence']:.2%}")
                if details['reasoning']:
                    lines.append(f"    Reasoning: {details['reasoning'][:100]}...")
            lines.append("")
        
        # Reasoning
        if output['reasoning']['summary']:
            lines.append("REASONING")
            lines.append("-" * 60)
            lines.append(output['reasoning']['summary'])
            lines.append("")
        
        # Evidence
        evidence = output['evidence']
        lines.append("EVIDENCE")
        lines.append("-" * 60)
        lines.append(f"Evidence Pages: {', '.join(map(str, evidence['page_citations']))}")
        lines.append(f"Evidence Sources: {', '.join(evidence['evidence_sources'])}")
        lines.append("")
        
        # PII Detections
        if output['pii_detections']:
            pii = output['pii_detections']
            lines.append("PII DETECTIONS")
            lines.append("-" * 60)
            lines.append(f"Total Detections: {pii['total_detections']}")
            lines.append(f"Types Found: {', '.join(pii['pii_types_found'])}")
            lines.append(f"Pages with PII: {', '.join(map(str, pii['pages_with_pii']))}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

