#!/usr/bin/env python3
"""
Stock Summary Agent - FastAPI Backend
Complete agent-based architecture(using LangChain) with LangGraph orchestration
"""

import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our agent system
from agents.coordinator import CoordinatorAgent
from models import AgentState, StockSummary, SystemConfig
from utils import setup_environment, log_info, log_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Summary Agent",
    description="AI-powered stock analysis platform with complete agent-based architecture",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator agent instance
coordinator_agent: Optional[CoordinatorAgent] = None


class AnalysisRequest(BaseModel):
    """Request model for stock analysis"""
    query: str = Field(..., description="Stock query to analyze")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Analysis options"
    )

class BatchAnalysisRequest(BaseModel):
    """Request model for batch stock analysis"""
    queries: List[str] = Field(..., description="List of stock queries")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Analysis options"
    )

class TickerValidationRequest(BaseModel):
    """Request model for ticker validation"""
    company_name: str = Field(..., description="Company name to validate")

class AnalysisResponse(BaseModel):
    """Response model for stock analysis"""
    success: bool
    company: Optional[str] = None
    ticker: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    success: bool
    results: List[AnalysisResponse]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TickerValidationResponse(BaseModel):
    """Response model for ticker validation"""
    valid: bool
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    agents_status: Dict[str, str]
    system_info: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the agent system on startup"""
    global coordinator_agent
    
    try:
        log_info("🚀 Starting Stock Summary Agent API")
        
        # Setup environment
        setup_environment()
        
        # Initialize coordinator agent
        log_info("🤖 Initializing agent system...")
        coordinator_agent = CoordinatorAgent()
        
        log_info("✅ Agent system initialized successfully")
        log_info("🌐 FastAPI server ready")
        
    except Exception as e:
        log_error(f"❌ Failed to initialize agent system: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    log_info("🛑 Shutting down Stock Summary Agent API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}s"
    )
    
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check agent system status
        agents_status = {}
        if coordinator_agent:
            agents_status = {
                "coordinator": "healthy",
                "research": "healthy",
                "analysis": "healthy",
                "sentiment": "healthy",
                "summarization": "healthy"
            }
        else:
            agents_status = {"system": "not_initialized"}
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            agents_status=agents_status,
            system_info={
                "version": "1.0.0",
                "architecture": "agent-based",
                "backend": "fastapi"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """Analyze a single stock"""
    try:
        if not coordinator_agent:
            raise HTTPException(status_code=500, detail="Agent system not initialized")
        
        log_info(f"📊 Analyzing stock: {request.query}")
        
        # Create session ID
        session_id = f"session_{int(time.time())}"
        
        # Process the query through the agent system
        result = await coordinator_agent.process_query(request.query, session_id)
        
        if result["success"]:
            # Format successful response
            analysis_data = result.get("final_summary", {})
            
            return AnalysisResponse(
                success=True,
                company=analysis_data.get("company_name", "Unknown"),
                ticker=analysis_data.get("ticker", "Unknown"),
                analysis={
                    "executive_summary": analysis_data.get("executive_summary", ""),
                    "technical_analysis": analysis_data.get("price_analysis", ""),
                    "sentiment_analysis": analysis_data.get("news_sentiment", ""),
                    "risk_assessment": analysis_data.get("risk_assessment", "")
                },
                metadata={
                    "processing_time": result.get("processing_time", 0),
                    "agents_used": result.get("agents_used", []),
                    "data_sources": result.get("data_sources", []),
                    "session_id": session_id
                }
            )
        else:
            return AnalysisResponse(
                success=False,
                error=result.get("error", "Analysis failed"),
                metadata={"session_id": session_id}
            )
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

@app.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_stocks(request: BatchAnalysisRequest):
    """Analyze multiple stocks concurrently"""
    try:
        if not coordinator_agent:
            raise HTTPException(status_code=500, detail="Agent system not initialized")
        
        log_info(f"📊 Batch analyzing {len(request.queries)} stocks")
        
        # Process queries concurrently
        tasks = []
        for query in request.queries:
            session_id = f"batch_{int(time.time())}_{len(tasks)}"
            tasks.append(coordinator_agent.process_query(query, session_id))
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        analysis_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                analysis_results.append(AnalysisResponse(
                    success=False,
                    error=f"Analysis failed: {str(result)}"
                ))
            else:
                if result["success"]:
                    analysis_data = result.get("final_summary", {})
                    analysis_results.append(AnalysisResponse(
                        success=True,
                        company=analysis_data.get("company_name", "Unknown"),
                        ticker=analysis_data.get("ticker", "Unknown"),
                        analysis={
                            "executive_summary": analysis_data.get("executive_summary", ""),
                            "technical_analysis": analysis_data.get("price_analysis", ""),
                            "sentiment_analysis": analysis_data.get("news_sentiment", ""),
                            "risk_assessment": analysis_data.get("risk_assessment", "")
                        },
                        metadata=result.get("metadata", {})
                    ))
                else:
                    analysis_results.append(AnalysisResponse(
                        success=False,
                        error=result.get("error", "Analysis failed")
                    ))
        
        successful_count = sum(1 for r in analysis_results if r.success)
        
        return BatchAnalysisResponse(
            success=True,
            results=analysis_results,
            metadata={
                "total_queries": len(request.queries),
                "successful": successful_count,
                "failed": len(request.queries) - successful_count
            }
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return BatchAnalysisResponse(
            success=False,
            results=[],
            metadata={"error": str(e)}
        )

@app.post("/validate-ticker", response_model=TickerValidationResponse)
async def validate_ticker(request: TickerValidationRequest):
    """Validate and resolve company ticker"""
    try:
        from services.ticker_lookup import TickerLookup
        
        lookup = TickerLookup()
        ticker = lookup.lookup_ticker(request.company_name)
        
        if ticker:
            company_name = lookup.get_company_name(ticker)
            return TickerValidationResponse(
                valid=True,
                ticker=ticker,
                company_name=company_name,
                suggestions=[]
            )
        else:
            suggestions = lookup.suggest_tickers(request.company_name, limit=5)
            return TickerValidationResponse(
                valid=False,
                ticker=None,
                company_name=None,
                suggestions=suggestions
            )
            
    except Exception as e:
        logger.error(f"Ticker validation failed: {str(e)}")
        return TickerValidationResponse(
            valid=False,
            suggestions=[],
        )

@app.get("/supported-companies")
async def get_supported_companies():
    """Get list of supported companies"""
    try:
        from services.ticker_lookup import TickerLookup
        
        lookup = TickerLookup()
        companies = list(lookup.common_tickers.keys())
        
        return {
            "success": True,
            "companies": sorted([company.title() for company in companies]),
            "count": len(companies)
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported companies: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "companies": [],
            "count": 0
        }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # Configuration
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    reload = os.getenv("FASTAPI_RELOAD", "false").lower() == "true"
    
    # Start server
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 