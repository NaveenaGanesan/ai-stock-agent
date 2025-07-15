"""
FastAPI request and response models for the Stock Summary Agent API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


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

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    agents_status: Dict[str, str]
    system_info: Dict[str, Any] 