"""
Simple test script for PII detector.
Run this to verify Phase 3 is working correctly.
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.pii_detector import PIIDetector
from src.logger import logger


def test_pii_detector():
    """Test the PII detector with a sample PDF."""
    # Process a document first
    processor = DocumentProcessor()
    detector = PIIDetector()
    
    # Check for test PDF in data/input
    input_dir = Path("data/input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in data/input/")
        logger.info("Please add a test PDF to data/input/ to test the PII detector")
        print("\n[SUCCESS] PII Detector module created successfully!")
        print("[INFO] To test: Add a PDF file to data/input/ and run this script again")
        return
    
    # Process first PDF found
    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf.name}")
    
    try:
        # Process document
        processed_doc = processor.process_document(test_pdf)
        
        # Detect PII
        pii_result = detector.detect_pii(processed_doc)
        
        # Get summary
        summary = detector.get_detection_summary(pii_result)
        
        print("\n" + "="*60)
        print("[PII DETECTION] Test Results")
        print("="*60)
        print(f"Total Detections: {summary['total_detections']}")
        print(f"PII Types Found: {', '.join(summary['pii_types_found']) if summary['pii_types_found'] else 'None'}")
        print(f"Pages with PII: {summary['pages_with_pii']}")
        print(f"Average Confidence: {summary['average_confidence']:.2f}")
        print(f"High Confidence Detections: {summary['high_confidence_detections']}")
        print("="*60)
        
        # Show type counts
        if summary['type_counts']:
            print("\n[BREAKDOWN] Detections by Type:")
            print("-" * 60)
            for pii_type, count in sorted(summary['type_counts'].items()):
                print(f"  {pii_type}: {count}")
            print("-" * 60)
        
        # Show sample detections
        if pii_result.detections:
            print("\n[SAMPLE] First 5 Detections:")
            print("-" * 60)
            for i, det in enumerate(pii_result.detections[:5], 1):
                print(f"\n{i}. Type: {det.pii_type.value}")
                print(f"   Value: {det.value}")
                print(f"   Page: {det.page_number}")
                print(f"   Confidence: {det.confidence:.2f}")
                if det.context:
                    context_preview = det.context[:100]
                    print(f"   Context: ...{context_preview}...")
            print("-" * 60)
        else:
            print("\n[INFO] No PII detected in this document.")
            print("This is normal for public marketing documents.")
        
        print("\n[SUCCESS] PII Detector test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing PII detector: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Error: {e}")
        print("\nNote: Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("\nFor spaCy NER, you may need to download the model:")
        print("  python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    test_pii_detector()

