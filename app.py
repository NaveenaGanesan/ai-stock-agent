#!/usr/bin/env python3
"""
Stock Summary Agent - FastAPI REST API
Provides FastAPI REST API with complete agent-based architecture
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our agent system
from agents.coordinator import CoordinatorAgent
from models import StockSummary
from utils import setup_environment, log_info, log_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# FASTAPI MODELS
# ===============================================================================

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

# ===============================================================================
# AGENT WRAPPER
# ===============================================================================

class StockSummaryAgent:
    """Main agent for stock analysis with FastAPI integration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Stock Summary Agent."""
        self.config = config or {}
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the agent system."""
        try:
            # Setup environment
            setup_environment()
            
            # Initialize coordinator
            self.coordinator = CoordinatorAgent(self.config)
            
            log_info("Stock Summary Agent initialized successfully")
            
        except Exception as e:
            log_error(f"Failed to initialize Stock Summary Agent: {str(e)}")
            raise
    
    async def analyze_stock(self, query: str) -> Dict[str, Any]:
        """Analyze a stock query and return comprehensive analysis."""
        try:
            log_info(f"Processing stock query: {query}")
            
            # Process query through coordinator
            result = await self.coordinator.process_query(query)
            
            if result.get("success", False):
                return self._format_success_response(result)
            else:
                return self._format_error_response(result)
                
        except Exception as e:
            log_error(f"Stock analysis failed: {str(e)}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "query": query,
                    "error_type": type(e).__name__
                }
            }
    
    def _format_success_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful analysis response."""
        final_summary = result.get("final_summary", {})
        
        # Extract core analysis data
        analysis_data = {
            "company_name": final_summary.get("company_name", "Unknown"),
            "ticker": final_summary.get("ticker", "Unknown"),
            "executive_summary": final_summary.get("executive_summary", ""),
            "price_analysis": final_summary.get("price_analysis", ""),
            "news_sentiment": final_summary.get("news_sentiment", ""),
            "technical_outlook": final_summary.get("technical_outlook", ""),
            "risk_assessment": final_summary.get("risk_assessment", ""),
            "confidence_level": final_summary.get("confidence_level", 0.0),
            "data_sources": final_summary.get("data_sources", [])
        }
        
        return {
            "success": True,
            "company": analysis_data["company_name"],
            "ticker": analysis_data["ticker"],
            "analysis": analysis_data,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "step_count": result.get("step_count", 0),
                "data_sources": analysis_data["data_sources"],
                "confidence_level": analysis_data["confidence_level"]
            }
        }
    
    def _format_error_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response."""
        errors = result.get("errors", ["Unknown error occurred"])
        
        return {
            "success": False,
            "error": "; ".join(errors),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "step_count": result.get("step_count", 0),
                "errors": errors
            }
        }
    
    async def get_supported_companies(self) -> List[str]:
        """Get list of supported companies."""
        try:
            # This could be expanded to return actual supported companies
            # For now, return a placeholder
            return [
                "Apple Inc. (AAPL)",
                "Microsoft Corporation (MSFT)",
                "Amazon.com Inc. (AMZN)",
                "Tesla Inc. (TSLA)",
                "Alphabet Inc. (GOOGL)"
            ]
        except Exception as e:
            log_error(f"Failed to get supported companies: {str(e)}")
            return []
    
    async def validate_company(self, company_name: str) -> Dict[str, Any]:
        """Validate a company name and return ticker if found."""
        try:
            # Use the ticker lookup agent for validation
            ticker_lookup = self.coordinator.ticker_lookup_agent
            result = await ticker_lookup.resolve_company_ticker(company_name)
            
            if result.get("success", False):
                return {
                    "valid": True,
                    "ticker": result.get("ticker"),
                    "company_name": result.get("company_name"),
                    "suggestions": []
                }
            else:
                # Try to get suggestions
                suggestions = await ticker_lookup.suggest_companies(company_name)
                return {
                    "valid": False,
                    "ticker": None,
                    "company_name": company_name,
                    "suggestions": suggestions
                }
                
        except Exception as e:
            log_error(f"Company validation failed: {str(e)}")
            return {
                "valid": False,
                "ticker": None,
                "company_name": company_name,
                "suggestions": [],
                "error": str(e)
            }

# ===============================================================================
# FASTAPI APPLICATION
# ===============================================================================

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

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent system on startup."""
    global agent
    try:
        log_info("🚀 Starting Stock Summary Agent API...")
        log_info("📋 Initializing agent system...")
        
        agent = StockSummaryAgent()
        
        log_info("✅ Agent system initialized successfully")
        log_info("🌐 API server ready to accept requests")
        
    except Exception as e:
        log_error(f"❌ Failed to initialize agent system: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    log_info("🔄 Shutting down Stock Summary Agent API...")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    log_info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# ===============================================================================
# API ENDPOINTS
# ===============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global agent
    
    agents_status = {}
    system_info = {}
    
    try:
        if agent and agent.coordinator:
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
            raise Exception("Agent not initialized")
            
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
    global agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        log_info(f"Received analysis request for: {request.query}")
        
        # Process the analysis request
        result = await agent.analyze_stock(request.query)
        
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
    global agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        log_info(f"Received batch analysis request for {len(request.queries)} queries")
        
        # Process all queries concurrently
        import asyncio
        
        # Create tasks for concurrent execution
        tasks = [agent.analyze_stock(query) for query in request.queries]
        
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

@app.post("/validate-ticker", response_model=TickerValidationResponse)
async def validate_ticker(request: TickerValidationRequest):
    """Validate a company name and return ticker information."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        log_info(f"Received ticker validation request for: {request.company_name}")
        
        result = await agent.validate_company(request.company_name)
        
        return TickerValidationResponse(
            valid=result.get("valid", False),
            ticker=result.get("ticker"),
            company_name=result.get("company_name"),
            suggestions=result.get("suggestions", [])
        )
        
    except Exception as e:
        log_error(f"Ticker validation endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ticker validation failed: {str(e)}")

@app.get("/supported-companies")
async def get_supported_companies():
    """Get list of supported companies."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        companies = await agent.get_supported_companies()
        return {"companies": companies}
        
    except Exception as e:
        log_error(f"Supported companies endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported companies: {str(e)}")

# ===============================================================================
# EXCEPTION HANDLERS
# ===============================================================================

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