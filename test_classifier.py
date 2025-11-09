"""
Simple test script for document classifier.
Run this to verify Phase 4 is working correctly.
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
from src.classifier import DocumentClassifier
from src.logger import logger


def test_classifier():
    """Test the document classifier with a sample PDF."""
    # Initialize components
    processor = DocumentProcessor()
    
    try:
        classifier = DocumentClassifier()
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        print(f"\n[ERROR] Failed to initialize classifier: {e}")
        print("\nNote: Make sure GEMINI_API_KEY is set in .env file")
        print("  If you don't have a Gemini API key, classification will use rules only")
        return
    
    # Check for test PDF in data/input
    input_dir = Path("data/input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in data/input/")
        logger.info("Please add a test PDF to data/input/ to test the classifier")
        print("\n[SUCCESS] Classifier module created successfully!")
        print("[INFO] To test: Add a PDF file to data/input/ and run this script again")
        return
    
    # Process first PDF found
    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf.name}")
    
    try:
        # Process document
        print("\n[STEP 1] Processing document...")
        processed_doc = processor.process_document(test_pdf)
        print(f"[SUCCESS] Document processed: {processed_doc.metadata.page_count} pages")
        
        # Classify document
        print("\n[STEP 2] Classifying document...")
        classification_result = classifier.classify(processed_doc)
        
        # Display results
        print("\n" + "="*60)
        print("[CLASSIFICATION] Results")
        print("="*60)
        print(f"Primary Category: {classification_result.primary_category.value if classification_result.primary_category else 'None'}")
        print(f"Overall Confidence: {classification_result.overall_confidence:.2f}")
        print(f"Categories: {[cat.value for cat in classification_result.categories]}")
        print(f"Evidence Pages: {classification_result.page_citations}")
        print("="*60)
        
        # Show category scores
        print("\n[SCORES] Category Confidence Scores:")
        print("-" * 60)
        for category_name, score in classification_result.category_scores.items():
            if score.confidence > 0:
                print(f"  {category_name}: {score.confidence:.2f}")
                if score.reasoning:
                    print(f"    Reasoning: {score.reasoning[:100]}...")
                if score.evidence_sources:
                    print(f"    Sources: {', '.join(score.evidence_sources)}")
        print("-" * 60)
        
        # Show PII detections
        if classification_result.pii_detections and classification_result.pii_detections.total_detections > 0:
            print(f"\n[PII] Detected {classification_result.pii_detections.total_detections} PII instances")
            print(f"  Types: {', '.join(classification_result.pii_detections.pii_types_found)}")
            print(f"  Pages with PII: {classification_result.pii_detections.pages_with_pii}")
        
        # Show reasoning
        if classification_result.reasoning:
            print(f"\n[REASONING] {classification_result.reasoning[:300]}...")
        
        # Show Gemini responses
        if classification_result.gemini_responses:
            print(f"\n[GEMINI] Processed {len(classification_result.gemini_responses)} pages with Gemini")
        
        print("\n[SUCCESS] Classification test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing classifier: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Error: {e}")


if __name__ == "__main__":
    test_classifier()

