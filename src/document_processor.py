"""
Document processing module for PDF parsing, OCR, and metadata extraction.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import easyocr

from src.config_loader import config
from src.logger import logger


@dataclass
class PageContent:
    """Represents content from a single page."""
    page_number: int
    text: str = ""
    images: List[Image.Image] = field(default_factory=list)
    ocr_text: str = ""
    has_text: bool = False
    is_scanned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """Document metadata information."""
    filename: str
    file_path: Path
    file_size: int
    page_count: int
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    producer: Optional[str] = None
    creator: Optional[str] = None
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Unified representation of a processed document."""
    metadata: DocumentMetadata
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""
    total_images: int = 0
    processing_errors: List[str] = field(default_factory=list)


class DocumentProcessor:
    """Processes PDF documents to extract text, images, and metadata."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.enable_ocr = config.get('document_processing.enable_ocr', True)
        self.ocr_language = config.get('document_processing.ocr_language', 'eng')
        self.max_file_size_mb = config.get('document_processing.max_file_size_mb', 50)
        self.supported_formats = config.get('document_processing.supported_formats', ['pdf'])
        
        # Initialize OCR readers
        self.easyocr_reader = None
        if self.enable_ocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}. Will use pytesseract only.")
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate that the file exists and meets requirements.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum ({self.max_file_size_mb} MB)"
        
        # Check file extension
        file_ext = file_path.suffix.lower().lstrip('.')
        if file_ext not in self.supported_formats:
            return False, f"Unsupported file format: {file_ext}. Supported: {self.supported_formats}"
        
        return True, None
    
    def extract_pdf_metadata(self, pdf_path: Path) -> DocumentMetadata:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentMetadata object
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Get PDF metadata
                pdf_info = pdf.metadata or {}
                
                # Extract dates
                creation_date = None
                mod_date = None
                
                if 'CreationDate' in pdf_info:
                    try:
                        creation_date = pdf_info['CreationDate']
                    except:
                        pass
                
                if 'ModDate' in pdf_info:
                    try:
                        mod_date = pdf_info['ModDate']
                    except:
                        pass
                
                metadata = DocumentMetadata(
                    filename=pdf_path.name,
                    file_path=pdf_path,
                    file_size=pdf_path.stat().st_size,
                    page_count=len(pdf.pages),
                    creation_date=creation_date,
                    modification_date=mod_date,
                    author=pdf_info.get('Author'),
                    title=pdf_info.get('Title'),
                    subject=pdf_info.get('Subject'),
                    producer=pdf_info.get('Producer'),
                    creator=pdf_info.get('Creator'),
                    pdf_metadata=pdf_info
                )
                
                logger.info(f"Extracted metadata for {pdf_path.name}: {metadata.page_count} pages")
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            # Return minimal metadata
            return DocumentMetadata(
                filename=pdf_path.name,
                file_path=pdf_path,
                file_size=pdf_path.stat().st_size,
                page_count=0
            )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[PageContent]:
        """
        Extract text from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PageContent objects
        """
        pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract text
                        text = page.extract_text() or ""
                        text = text.strip()
                        
                        page_content = PageContent(
                            page_number=page_num,
                            text=text,
                            has_text=len(text) > 0,
                            is_scanned=len(text) < 50  # Heuristic: very little text = likely scanned
                        )
                        
                        pages.append(page_content)
                        logger.debug(f"Extracted text from page {page_num}: {len(text)} characters")
                        
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        pages.append(PageContent(page_number=page_num))
                        
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            raise
        
        return pages
    
    def extract_images_from_pdf(self, pdf_path: Path) -> List[Image.Image]:
        """
        Extract images from PDF pages using pdf2image.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Image objects (one per page)
        """
        images = []
        
        try:
            # Convert PDF pages to images
            # Note: This requires poppler to be installed
            pdf_images = convert_from_path(
                pdf_path,
                dpi=200,  # Good balance between quality and file size
                fmt='RGB'
            )
            
            images.extend(pdf_images)
            logger.info(f"Extracted {len(images)} page images from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF {pdf_path}: {e}")
            logger.warning("Image extraction failed. This may require poppler-utils to be installed.")
            # Return empty list - document can still be processed with text only
        
        return images
    
    def perform_ocr(self, image: Image.Image, page_num: int) -> str:
        """
        Perform OCR on an image using pytesseract or easyocr.
        
        Args:
            image: PIL Image object
            page_num: Page number for logging
            
        Returns:
            Extracted text from OCR
        """
        ocr_text = ""
        
        if not self.enable_ocr:
            return ocr_text
        
        try:
            # Try pytesseract first (faster, lighter)
            try:
                ocr_text = pytesseract.image_to_string(image, lang=self.ocr_language)
                logger.debug(f"OCR (pytesseract) completed for page {page_num}: {len(ocr_text)} characters")
            except Exception as e:
                logger.warning(f"pytesseract OCR failed for page {page_num}: {e}")
                
                # Fallback to easyocr
                if self.easyocr_reader:
                    try:
                        results = self.easyocr_reader.readtext(image)
                        ocr_text = "\n".join([result[1] for result in results])
                        logger.debug(f"OCR (easyocr) completed for page {page_num}: {len(ocr_text)} characters")
                    except Exception as e2:
                        logger.error(f"easyocr OCR also failed for page {page_num}: {e2}")
        
        except Exception as e:
            logger.error(f"OCR error for page {page_num}: {e}")
        
        return ocr_text.strip()
    
    def process_document(self, file_path: Path) -> ProcessedDocument:
        """
        Main method to process a document and extract all content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessedDocument object with all extracted content
        """
        # Validate file
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        logger.info(f"Processing document: {file_path.name}")
        
        # Extract metadata
        metadata = self.extract_pdf_metadata(file_path)
        
        # Extract text from PDF
        pages = self.extract_text_from_pdf(file_path)
        
        # Extract page images
        page_images = self.extract_images_from_pdf(file_path)
        
        # Match images to pages and perform OCR if needed
        processing_errors = []
        total_images = 0
        
        for i, page_content in enumerate(pages):
            # Add image if available
            if i < len(page_images):
                page_content.images.append(page_images[i])
                total_images += 1
            
            # Perform OCR if page appears to be scanned (little or no text)
            if page_content.is_scanned or (not page_content.has_text and len(page_content.images) > 0):
                try:
                    if page_content.images:
                        ocr_text = self.perform_ocr(page_content.images[0], page_content.page_number)
                        page_content.ocr_text = ocr_text
                        if ocr_text:
                            # Combine OCR text with existing text
                            page_content.text = f"{page_content.text}\n{ocr_text}".strip()
                            page_content.has_text = True
                            page_content.is_scanned = True
                except Exception as e:
                    error_msg = f"OCR error on page {page_content.page_number}: {e}"
                    logger.warning(error_msg)
                    processing_errors.append(error_msg)
        
        # Combine all text
        full_text = "\n\n".join([
            f"--- Page {p.page_number} ---\n{p.text}"
            for p in pages if p.text
        ])
        
        # Create processed document
        processed_doc = ProcessedDocument(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            total_images=total_images,
            processing_errors=processing_errors
        )
        
        logger.info(
            f"Document processing complete: {len(pages)} pages, "
            f"{len(full_text)} characters, {total_images} images"
        )
        
        return processed_doc
    
    def get_page_summary(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """
        Get a summary of the processed document.
        
        Args:
            processed_doc: ProcessedDocument object
            
        Returns:
            Dictionary with document summary
        """
        return {
            "filename": processed_doc.metadata.filename,
            "page_count": processed_doc.metadata.page_count,
            "total_characters": len(processed_doc.full_text),
            "total_images": processed_doc.total_images,
            "pages_with_text": sum(1 for p in processed_doc.pages if p.has_text),
            "scanned_pages": sum(1 for p in processed_doc.pages if p.is_scanned),
            "processing_errors": len(processed_doc.processing_errors)
        }

