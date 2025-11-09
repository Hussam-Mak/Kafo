"""
FastAPI backend for document classification.
"""

import sys
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.classifier import DocumentClassifier
from src.output_formatter import OutputFormatter
from src.logger import logger
from src.config_loader import config


# Pydantic models for API
class ClassificationResponse(BaseModel):
    """API response model for classification."""
    status: str
    document_metadata: dict
    classification: dict
    reasoning: dict
    evidence: dict
    pii_detections: Optional[dict] = None
    processing_info: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    gemini_enabled: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str
    error: str
    message: str


# Initialize FastAPI app
app = FastAPI(
    title="AI Document Classifier API",
    description="AI-Powered Regulatory Document Classification System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
processor = None
classifier = None
formatter = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global processor, classifier, formatter
    try:
        processor = DocumentProcessor()
        classifier = DocumentClassifier()
        formatter = OutputFormatter()
        logger.info("API components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API components: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    gemini_enabled = classifier.gemini_client is not None if classifier else False
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gemini_enabled=gemini_enabled
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    gemini_enabled = classifier.gemini_client is not None if classifier else False
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gemini_enabled=gemini_enabled
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(
    file: UploadFile = File(...),
    save_output: bool = True
):
    """
    Classify a document.
    
    Args:
        file: PDF file to classify
        save_output: Whether to save JSON output to disk
        
    Returns:
        Classification result
    """
    if not processor or not classifier or not formatter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again."
        )
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Save uploaded file temporarily
        temp_path = Path("data/input") / file.filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Process document
        processed_doc = processor.process_document(temp_path)
        
        # Classify document
        classification_result = classifier.classify(processed_doc)
        
        # Format output
        output = formatter.format_classification_result(
            classification_result,
            processed_doc
        )
        
        # Save JSON output if requested
        if save_output:
            formatter.save_json_output(output)
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
        
        # Return API-formatted response
        api_response = formatter.format_for_api(output)
        api_response["document_metadata"] = output["document_metadata"]
        api_response["reasoning"] = output["reasoning"]
        api_response["pii_detections"] = output["pii_detections"]
        api_response["processing_info"] = output["processing_info"]
        
        return ClassificationResponse(**api_response)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/classify/batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    save_output: bool = True
):
    """
    Classify multiple documents in batch.
    
    Args:
        files: List of PDF files to classify
        save_output: Whether to save JSON output to disk
        
    Returns:
        List of classification results
    """
    if not processor or not classifier or not formatter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again."
        )
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.filename.endswith('.pdf'):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "Only PDF files are supported"
                })
                continue
            
            # Save uploaded file temporarily
            temp_path = Path("data/input") / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process document
            processed_doc = processor.process_document(temp_path)
            
            # Classify document
            classification_result = classifier.classify(processed_doc)
            
            # Format output
            output = formatter.format_classification_result(
                classification_result,
                processed_doc
            )
            
            # Save JSON output if requested
            if save_output:
                formatter.save_json_output(output)
            
            # Clean up temp file
            try:
                temp_path.unlink()
            except:
                pass
            
            # Format for API
            api_response = formatter.format_for_api(output)
            api_response["filename"] = file.filename
            api_response["status"] = "success"
            results.append(api_response)
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results, "total": len(files), "processed": len([r for r in results if r.get("status") == "success"])}


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()

