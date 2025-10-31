#!/usr/bin/env python3
"""
FastAPI wrapper for Smart Byggefakta PDF Scraper
Provides REST API endpoint for PDF upload and data extraction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pathlib import Path
from dataclasses import asdict
import logging

# Import our scraper
from smart_scraper import parse_pdf, ParsedDocument

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Byggefakta PDF Scraper API",
    description="Extract contacts, projects, and company info from Smart Byggefakta PDFs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Smart Byggefakta PDF Scraper",
        "version": "1.0.0",
        "endpoints": {
            "POST /parse": "Upload PDF and extract data",
            "POST /parse-multiple": "Upload multiple PDFs",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    return {"status": "healthy"}

@app.post("/parse")
async def parse_pdf_endpoint(file: UploadFile = File(...)):
    """
    Upload a Smart Byggefakta PDF and extract structured data
    
    Returns:
    - company_info: Company details (name, CVR, contact info)
    - contacts: List of contacts with names, phones, emails, roles
    - projects: List of projects with budgets, regions, stages
    - tenders: List of tenders
    """
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    logger.info(f"Received PDF: {file.filename}")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"Saved to temporary file: {tmp_path}")
        
        # Parse the PDF
        doc = parse_pdf(tmp_path)
        
        # Convert to dict
        result = asdict(doc)
        
        # Add metadata
        result['metadata'] = {
            'filename': file.filename,
            'extraction_success': True,
            'total_contacts': len(doc.contacts),
            'total_projects': len(doc.projects),
            'total_tenders': len(doc.tenders)
        }
        
        logger.info(f"Extraction complete: {len(doc.contacts)} contacts, {len(doc.projects)} projects")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    
    finally:
        # Cleanup temporary file
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")

@app.post("/parse-multiple")
async def parse_multiple_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload multiple Smart Byggefakta PDFs and extract data from all
    
    Returns list of extraction results
    """
    
    results = []
    errors = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            errors.append({
                'filename': file.filename,
                'error': 'File must be a PDF'
            })
            continue
        
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            logger.info(f"Processing: {file.filename}")
            
            # Parse the PDF
            doc = parse_pdf(tmp_path)
            
            # Convert to dict and add metadata
            result = asdict(doc)
            result['metadata'] = {
                'filename': file.filename,
                'extraction_success': True,
                'total_contacts': len(doc.contacts),
                'total_projects': len(doc.projects),
                'total_tenders': len(doc.tenders)
            }
            
            results.append(result)
            
            # Cleanup
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Error parsing {file.filename}: {e}")
            errors.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return JSONResponse(content={
        'results': results,
        'errors': errors,
        'total_processed': len(results),
        'total_failed': len(errors)
    })

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "status": "operational",
        "features": {
            "contact_extraction": True,
            "project_extraction": True,
            "tender_extraction": True,
            "company_info_extraction": True,
            "multi_pdf_support": True
        },
        "supported_formats": ["PDF"],
        "supported_sections": ["KONTAKTER", "PROJEKTER", "UDBUD", "OPLYSNINGER"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
