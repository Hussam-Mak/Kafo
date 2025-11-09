"""
CLI entry point for the AI Document Classifier.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.classifier import DocumentClassifier
from src.output_formatter import OutputFormatter
from src.logger import logger


def classify_document(file_path: Path, output_dir: Path = None, save_json: bool = True):
    """Classify a single document."""
    try:
        # Initialize components
        processor = DocumentProcessor()
        classifier = DocumentClassifier()
        formatter = OutputFormatter()
        
        # Process document
        print(f"[INFO] Processing document: {file_path.name}")
        processed_doc = processor.process_document(file_path)
        print(f"[SUCCESS] Document processed: {processed_doc.metadata.page_count} pages")
        
        # Classify document
        print(f"[INFO] Classifying document...")
        classification_result = classifier.classify(processed_doc)
        print(f"[SUCCESS] Classification complete")
        
        # Format output
        output = formatter.format_classification_result(classification_result, processed_doc)
        
        # Display results
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Primary Category: {output['classification']['primary_category']}")
        print(f"Overall Confidence: {output['classification']['overall_confidence']:.2%}")
        print(f"Categories: {', '.join(output['classification']['categories'])}")
        print(f"Evidence Pages: {output['evidence']['page_citations']}")
        if output['pii_detections']:
            print(f"PII Detections: {output['pii_detections']['total_detections']}")
        print("="*60)
        
        # Save JSON output
        if save_json:
            json_path = formatter.save_json_output(output, output_dir=output_dir)
            print(f"\n[SUCCESS] JSON output saved to: {json_path}")
        
        # Show human-readable summary
        print("\n" + formatter.create_human_readable_summary(output))
        
        return output
    
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        print(f"\n[ERROR] Failed to classify document: {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_batch(input_dir: Path, output_dir: Path = None, save_json: bool = True):
    """Classify multiple documents in batch."""
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(pdf_files)} PDF file(s) to process")
    
    # Initialize components once
    processor = DocumentProcessor()
    classifier = DocumentClassifier()
    formatter = OutputFormatter()
    
    results = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        try:
            # Process document
            processed_doc = processor.process_document(pdf_file)
            
            # Classify document
            classification_result = classifier.classify(processed_doc)
            
            # Format output
            output = formatter.format_classification_result(classification_result, processed_doc)
            
            # Save JSON output
            if save_json:
                formatter.save_json_output(output, output_dir=output_dir)
            
            # Display summary
            primary = output['classification']['primary_category']
            confidence = output['classification']['overall_confidence']
            print(f"  -> {primary} ({confidence:.2%})")
            
            results.append({
                "filename": pdf_file.name,
                "status": "success",
                "category": primary,
                "confidence": confidence
            })
        
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            print(f"  -> ERROR: {e}")
            results.append({
                "filename": pdf_file.name,
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {len(pdf_files)}")
    print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results if r['status'] == 'error'])}")
    print("="*60)
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Regulatory Document Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single document
  python main.py classify document.pdf
  
  # Classify a single document and save JSON
  python main.py classify document.pdf --output results/
  
  # Batch process all PDFs in a directory
  python main.py batch data/input/
  
  # Start API server
  python main.py api --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a single document")
    classify_parser.add_argument("file", type=Path, help="Path to PDF file")
    classify_parser.add_argument("--output", "-o", type=Path, help="Output directory for JSON files")
    classify_parser.add_argument("--no-json", action="store_true", help="Don't save JSON output")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple documents")
    batch_parser.add_argument("input_dir", type=Path, help="Directory containing PDF files")
    batch_parser.add_argument("--output", "-o", type=Path, help="Output directory for JSON files")
    batch_parser.add_argument("--no-json", action="store_true", help="Don't save JSON output")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "classify":
        if not args.file.exists():
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        
        classify_document(
            args.file,
            output_dir=args.output,
            save_json=not args.no_json
        )
    
    elif args.command == "batch":
        if not args.input_dir.exists() or not args.input_dir.is_dir():
            print(f"[ERROR] Directory not found: {args.input_dir}")
            sys.exit(1)
        
        classify_batch(
            args.input_dir,
            output_dir=args.output,
            save_json=not args.no_json
        )
    
    elif args.command == "api":
        print(f"[INFO] Starting API server on {args.host}:{args.port}")
        print(f"[INFO] API documentation available at http://{args.host}:{args.port}/docs")
        from src.api import run_api_server
        run_api_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

