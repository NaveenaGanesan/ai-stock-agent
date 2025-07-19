#!/usr/bin/env python3
"""
Stock Summary Agent - FastAPI REST API
Provides FastAPI REST API endpoints for stock analysis
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.stock_summary_service import StockSummaryService
from models import (
    StockSummary,
    AnalysisRequest,
    BatchAnalysisRequest,
    HealthResponse,
    AnalysisResponse,
    BatchAnalysisResponse,
)
from utils import log_info, log_error

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Stock Summary Agent",
    description="Advanced AI-powered stock analysis and summary generation",
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

# Global service instance
service = None

@app.on_event("startup")
async def startup_event():
    """Initialize the service layer on startup."""
    global service
    try:
        log_info("Starting Stock Summary Agent API...")
        log_info("Initializing service layer...")
        
        service = StockSummaryService()
        
        log_info("Service layer initialized successfully")
        log_info("API server ready to accept requests")
        
    except Exception as e:
        log_error(f"Failed to initialize service layer: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    log_info("Shutting down Stock Summary Agent API...")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    log_info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global service
    
    agents_status = {}
    system_info = {}
    
    try:
        if service and service.coordinator:
            # Check agent status
            agents_status = {
                "coordinator": "healthy",
                "ticker_lookup": "healthy",
                "research": "healthy",
                "analysis": "healthy",
                "sentiment": "healthy",
                "summarization": "healthy"
            }
            
            # System info
            system_info = {
                "python_version": "3.8+",
                "langchain_available": True,
                "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
                "environment": os.getenv('ENVIRONMENT', 'development')
            }
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                agents_status=agents_status,
                system_info=system_info
            )
        else:
            raise Exception("Service not initialized")
            
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            agents_status={"error": str(e)},
            system_info={"error": "System check failed"}
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """Analyze a stock query and return comprehensive analysis."""
    global service
    
    if not service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        log_info(f"Received analysis request for: {request.query}")
        
        # Process the analysis request
        result = await service.analyze_stock(request.query)
        
        if result.get("success", False):
            return AnalysisResponse(
                success=True,
                company=result.get("company"),
                ticker=result.get("ticker"),
                analysis=result.get("analysis"),
                metadata=result.get("metadata", {})
            )
        else:
            return AnalysisResponse(
                success=False,
                error=result.get("error", "Analysis failed"),
                metadata=result.get("metadata", {})
            )
            
    except Exception as e:
        log_error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_stocks(request: BatchAnalysisRequest):
    """Analyze multiple stock queries concurrently."""
    global service
    
    if not service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        log_info(f"Received batch analysis request for {len(request.queries)} queries")
        
        # Create tasks for concurrent execution
        tasks = [service.analyze_stock(query) for query in request.queries]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AnalysisResponse(
                    success=False,
                    error=str(result),
                    metadata={"query": request.queries[i]}
                ))
            else:
                processed_results.append(AnalysisResponse(
                    success=result.get("success", False),
                    company=result.get("company"),
                    ticker=result.get("ticker"),
                    analysis=result.get("analysis"),
                    metadata=result.get("metadata", {}),
                    error=result.get("error")
                ))
        
        return BatchAnalysisResponse(
            success=True,
            results=processed_results,
            metadata={
                "total_queries": len(request.queries),
                "processed_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        log_error(f"Batch analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    ) 