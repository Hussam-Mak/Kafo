"""
Simple test script for document processor.
Run this to verify Phase 2 is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.logger import logger


def test_document_processor():
    """Test the document processor with a sample PDF."""
    processor = DocumentProcessor()
    
    # Check for test PDF in data/input
    input_dir = Path("data/input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in data/input/")
        logger.info("Please add a test PDF to data/input/ to test the processor")
        print("\n✅ Document Processor module created successfully!")
        print("📝 To test: Add a PDF file to data/input/ and run this script again")
        return
    
    # Process first PDF found
    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf.name}")
    
    try:
        # Process document
        processed_doc = processor.process_document(test_pdf)
        
        # Get summary
        summary = processor.get_page_summary(processed_doc)
        
        print("\n" + "="*60)
        print("📄 Document Processing Test Results")
        print("="*60)
        print(f"Filename: {summary['filename']}")
        print(f"Pages: {summary['page_count']}")
        print(f"Total Characters: {summary['total_characters']:,}")
        print(f"Total Images: {summary['total_images']}")
        print(f"Pages with Text: {summary['pages_with_text']}")
        print(f"Scanned Pages: {summary['scanned_pages']}")
        print(f"Processing Errors: {summary['processing_errors']}")
        print("="*60)
        
        # Show sample text from first page
        if processed_doc.pages:
            first_page = processed_doc.pages[0]
            if first_page.text:
                preview = first_page.text[:200]
                print(f"\n📝 First Page Text Preview (first 200 chars):")
                print("-" * 60)
                print(preview + ("..." if len(first_page.text) > 200 else ""))
                print("-" * 60)
        
        print("\n✅ Document Processor test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        print(f"\n❌ Error: {e}")
        print("\nNote: Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("\nFor image extraction, you may also need poppler-utils:")
        print("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
        print("  Linux: sudo apt-get install poppler-utils")
        print("  Mac: brew install poppler")


if __name__ == "__main__":
    test_document_processor()

