"""
Test script for output formatter.
Run this to verify Phase 6 is working correctly.
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
from src.output_formatter import OutputFormatter
from src.logger import logger


def test_output_formatter():
    """Test the output formatter with a sample PDF."""
    # Initialize components
    processor = DocumentProcessor()
    
    try:
        classifier = DocumentClassifier()
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        print(f"\n[ERROR] Failed to initialize classifier: {e}")
        return
    
    formatter = OutputFormatter()
    
    # Check for test PDF in data/input
    input_dir = Path("data/input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in data/input/")
        print("\n[SUCCESS] Output Formatter module created successfully!")
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
        print(f"[SUCCESS] Classification complete: {classification_result.primary_category.value if classification_result.primary_category else 'None'}")
        
        # Format output
        print("\n[STEP 3] Formatting output...")
        output = formatter.format_classification_result(classification_result, processed_doc)
        print("[SUCCESS] Output formatted")
        
        # Save JSON
        print("\n[STEP 4] Saving JSON output...")
        json_path = formatter.save_json_output(output)
        print(f"[SUCCESS] JSON saved to: {json_path}")
        
        # Display summary
        print("\n" + "="*60)
        print("[OUTPUT] Classification Summary")
        print("="*60)
        print(f"Primary Category: {output['classification']['primary_category']}")
        print(f"Overall Confidence: {output['classification']['overall_confidence']:.2%}")
        print(f"Categories: {', '.join(output['classification']['categories'])}")
        print(f"Evidence Pages: {output['evidence']['page_citations']}")
        print(f"PII Detections: {output['pii_detections']['total_detections'] if output['pii_detections'] else 0}")
        print("="*60)
        
        # Show human-readable summary
        print("\n[HUMAN READABLE] Summary:")
        print("-" * 60)
        summary = formatter.create_human_readable_summary(output)
        print(summary)
        
        print("\n[SUCCESS] Output formatter test completed successfully!")
        print(f"[INFO] Check the JSON file at: {json_path}")
        
    except Exception as e:
        logger.error(f"Error testing output formatter: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Error: {e}")


if __name__ == "__main__":
    test_output_formatter()

