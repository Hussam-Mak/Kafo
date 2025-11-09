"""
PII Detection module using regex patterns and spaCy NER.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.config_loader import config
from src.logger import logger
from src.document_processor import ProcessedDocument, PageContent


class PIIType(Enum):
    """Types of PII that can be detected."""
    SSN = "SSN"
    CREDIT_CARD = "Credit Card"
    PHONE = "Phone Number"
    EMAIL = "Email Address"
    BANK_ACCOUNT = "Bank Account"
    PERSON_NAME = "Person Name"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    DATE_OF_BIRTH = "Date of Birth"
    DRIVER_LICENSE = "Driver's License"
    PASSPORT = "Passport Number"
    IP_ADDRESS = "IP Address"
    MAC_ADDRESS = "MAC Address"


@dataclass
class PIIDetection:
    """Represents a single PII detection."""
    pii_type: PIIType
    value: str  # Masked or partial value for security
    confidence: float  # 0.0 to 1.0
    page_number: int
    start_position: int  # Character position in page text
    end_position: int
    context: str = ""  # Surrounding text for context
    raw_value: str = ""  # Original unmasked value (for internal use only)


@dataclass
class PIIDetectionResult:
    """Results of PII detection on a document."""
    detections: List[PIIDetection] = field(default_factory=list)
    total_detections: int = 0
    pii_types_found: List[str] = field(default_factory=list)
    pages_with_pii: List[int] = field(default_factory=list)
    confidence_score: float = 0.0  # Overall confidence in PII detection


class PIIDetector:
    """Detects personally identifiable information in documents."""
    
    def __init__(self):
        """Initialize the PII detector."""
        self.enable_regex = config.get('pii_detection.enable_regex', True)
        self.enable_ner = config.get('pii_detection.enable_ner', True)
        self.patterns_config = config.get('pii_detection.patterns', {})
        
        # Initialize spaCy NER if available
        self.nlp = None
        if self.enable_ner and SPACY_AVAILABLE:
            try:
                # Try to load the English model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NER model loaded successfully")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                logger.warning("NER will be disabled. Only regex patterns will be used.")
                self.enable_ner = False
        elif self.enable_ner and not SPACY_AVAILABLE:
            logger.warning("spaCy not installed. NER will be disabled.")
            self.enable_ner = False
        
        # Compile regex patterns
        self.regex_patterns = self._compile_regex_patterns()
    
    def _compile_regex_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = {}
        
        # SSN pattern (XXX-XX-XXXX or XXX XX XXXX)
        if self.patterns_config.get('ssn', True):
            patterns[PIIType.SSN] = re.compile(
                r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
                re.IGNORECASE
            )
        
        # Credit card pattern (supports major card types)
        if self.patterns_config.get('credit_card', True):
            # Matches 13-19 digit numbers that could be credit cards
            # Note: This is a basic pattern - real validation requires Luhn algorithm
            patterns[PIIType.CREDIT_CARD] = re.compile(
                r'\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b',
                re.IGNORECASE
            )
        
        # Phone number pattern (various formats)
        if self.patterns_config.get('phone', True):
            patterns[PIIType.PHONE] = re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                re.IGNORECASE
            )
        
        # Email pattern
        if self.patterns_config.get('email', True):
            patterns[PIIType.EMAIL] = re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            )
        
        # Bank account pattern (generic - varies by country)
        if self.patterns_config.get('bank_account', True):
            # Matches 8-17 digit account numbers
            patterns[PIIType.BANK_ACCOUNT] = re.compile(
                r'\b(?:account|acct|acc)[\s:]*#?[\s-]?\d{8,17}\b',
                re.IGNORECASE
            )
        
        # Driver's License (US format - varies by state)
        patterns[PIIType.DRIVER_LICENSE] = re.compile(
            r'\b(?:DL|drivers?[\s-]?license|license[\s-]?number)[\s:]*#?[\s-]?[A-Z0-9]{6,12}\b',
            re.IGNORECASE
        )
        
        # Passport number (varies by country)
        patterns[PIIType.PASSPORT] = re.compile(
            r'\b(?:passport[\s-]?number|passport[\s-]?#)[\s:]*#?[\s-]?[A-Z0-9]{6,12}\b',
            re.IGNORECASE
        )
        
        # IP Address
        patterns[PIIType.IP_ADDRESS] = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        )
        
        # MAC Address
        patterns[PIIType.MAC_ADDRESS] = re.compile(
            r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'
        )
        
        # Date of Birth (various formats)
        patterns[PIIType.DATE_OF_BIRTH] = re.compile(
            r'\b(?:DOB|date[\s-]?of[\s-]?birth|born)[\s:]*[\s-]?(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            re.IGNORECASE
        )
        
        return patterns
    
    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask a PII value for display."""
        if pii_type == PIIType.SSN:
            # Show only last 4 digits: XXX-XX-1234
            if len(value.replace('-', '').replace(' ', '')) == 9:
                digits = re.sub(r'[^\d]', '', value)
                return f"XXX-XX-{digits[-4:]}"
            return "XXX-XX-XXXX"
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Show only last 4 digits: XXXX-XXXX-XXXX-1234
            digits = re.sub(r'[^\d]', '', value)
            if len(digits) >= 4:
                return f"{'X' * (len(digits) - 4)}-{digits[-4:]}"
            return "XXXX-XXXX-XXXX-XXXX"
        
        elif pii_type == PIIType.PHONE:
            # Show only last 4 digits: (XXX) XXX-1234
            digits = re.sub(r'[^\d]', '', value)
            if len(digits) >= 4:
                return f"(XXX) XXX-{digits[-4:]}"
            return "(XXX) XXX-XXXX"
        
        elif pii_type == PIIType.EMAIL:
            # Show only domain: xxxxx@example.com
            if '@' in value:
                local, domain = value.split('@', 1)
                return f"{'X' * min(len(local), 5)}@{domain}"
            return "xxxxx@xxxxx.xxx"
        
        elif pii_type == PIIType.BANK_ACCOUNT:
            # Show only last 4 digits
            digits = re.sub(r'[^\d]', '', value)
            if len(digits) >= 4:
                return f"XXXX-{digits[-4:]}"
            return "XXXX-XXXX"
        
        else:
            # Generic masking
            if len(value) > 8:
                return f"{value[:2]}...{value[-2:]}"
            return "XXXX"
    
    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = re.sub(r'[^\d]', '', number)
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0
        
        return luhn_check(digits)
    
    def _detect_regex_pii(self, text: str, page_num: int) -> List[PIIDetection]:
        """Detect PII using regex patterns."""
        detections = []
        
        for pii_type, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                # Get context (50 chars before and after)
                context_start = max(0, start_pos - 50)
                context_end = min(len(text), end_pos + 50)
                context = text[context_start:context_end].strip()
                
                # Calculate confidence
                confidence = 0.8  # Base confidence for regex matches
                
                # Special validation for credit cards
                if pii_type == PIIType.CREDIT_CARD:
                    if self._validate_credit_card(value):
                        confidence = 0.95
                    else:
                        confidence = 0.5  # Lower confidence if Luhn check fails
                
                # Lower confidence for SSN if it doesn't match common format
                if pii_type == PIIType.SSN:
                    digits = re.sub(r'[^\d]', '', value)
                    if len(digits) != 9:
                        confidence = 0.3
                
                # Create detection
                detection = PIIDetection(
                    pii_type=pii_type,
                    value=self._mask_value(value, pii_type),
                    confidence=confidence,
                    page_number=page_num,
                    start_position=start_pos,
                    end_position=end_pos,
                    context=context,
                    raw_value=value
                )
                
                detections.append(detection)
        
        return detections
    
    def _detect_ner_pii(self, text: str, page_num: int) -> List[PIIDetection]:
        """Detect PII using spaCy NER."""
        detections = []
        
        if not self.nlp or not self.enable_ner:
            return detections
        
        try:
            doc = self.nlp(text)
            
            # Map spaCy entities to PII types
            entity_mapping = {
                'PERSON': PIIType.PERSON_NAME,
                'ORG': PIIType.ORGANIZATION,
                'GPE': PIIType.LOCATION,  # Geopolitical entity (countries, cities, states)
                'LOC': PIIType.LOCATION,   # Non-GPE locations
            }
            
            for ent in doc.ents:
                if ent.label_ in entity_mapping:
                    pii_type = entity_mapping[ent.label_]
                    
                    # Get context
                    context_start = max(0, ent.start_char - 50)
                    context_end = min(len(text), ent.end_char + 50)
                    context = text[context_start:context_end].strip()
                    
                    # Calculate confidence based on spaCy's confidence
                    # spaCy doesn't provide explicit confidence, so we use label reliability
                    confidence = 0.7  # Base confidence for NER
                    if ent.label_ == 'PERSON':
                        confidence = 0.75
                    elif ent.label_ in ['ORG', 'GPE', 'LOC']:
                        confidence = 0.7
                    
                    detection = PIIDetection(
                        pii_type=pii_type,
                        value=ent.text,  # Don't mask names/locations for context
                        confidence=confidence,
                        page_number=page_num,
                        start_position=ent.start_char,
                        end_position=ent.end_char,
                        context=context,
                        raw_value=ent.text
                    )
                    
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error in NER detection: {e}")
        
        return detections
    
    def detect_pii(self, document: ProcessedDocument) -> PIIDetectionResult:
        """
        Detect PII in a processed document.
        
        Args:
            document: ProcessedDocument object
            
        Returns:
            PIIDetectionResult with all detected PII
        """
        all_detections = []
        pages_with_pii = set()
        pii_types_found = set()
        
        logger.info(f"Starting PII detection on {document.metadata.filename}")
        
        # Process each page
        for page in document.pages:
            if not page.text:
                continue
            
            page_detections = []
            
            # Regex-based detection
            if self.enable_regex:
                regex_detections = self._detect_regex_pii(page.text, page.page_number)
                page_detections.extend(regex_detections)
            
            # NER-based detection
            if self.enable_ner:
                ner_detections = self._detect_ner_pii(page.text, page.page_number)
                page_detections.extend(ner_detections)
            
            # Add to results
            if page_detections:
                all_detections.extend(page_detections)
                pages_with_pii.add(page.page_number)
                for det in page_detections:
                    pii_types_found.add(det.pii_type.value)
        
        # Calculate overall confidence score
        if all_detections:
            avg_confidence = sum(d.confidence for d in all_detections) / len(all_detections)
        else:
            avg_confidence = 0.0
        
        result = PIIDetectionResult(
            detections=all_detections,
            total_detections=len(all_detections),
            pii_types_found=sorted(list(pii_types_found)),
            pages_with_pii=sorted(list(pages_with_pii)),
            confidence_score=avg_confidence
        )
        
        logger.info(
            f"PII detection complete: {result.total_detections} detections, "
            f"{len(result.pii_types_found)} types found, "
            f"{len(result.pages_with_pii)} pages with PII"
        )
        
        return result
    
    def get_detection_summary(self, result: PIIDetectionResult) -> Dict[str, Any]:
        """Get a summary of PII detection results."""
        # Count by type
        type_counts = {}
        for det in result.detections:
            pii_type = det.pii_type.value
            type_counts[pii_type] = type_counts.get(pii_type, 0) + 1
        
        return {
            "total_detections": result.total_detections,
            "pii_types_found": result.pii_types_found,
            "type_counts": type_counts,
            "pages_with_pii": result.pages_with_pii,
            "average_confidence": result.confidence_score,
            "high_confidence_detections": sum(1 for d in result.detections if d.confidence >= 0.8)
        }

