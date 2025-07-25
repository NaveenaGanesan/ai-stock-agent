"""
Stock Summary Service - API service layer for stock analysis routing.
Handles API requests and routes them to the appropriate agents.
"""

from datetime import datetime
from typing import Dict, List, Any

from agents.coordinator_agent import CoordinatorAgent
from utils import log_info, log_error


class StockSummaryService:
    """API service layer for stock analysis with routing logic."""
    
    def __init__(self):
        """Initialize the Stock Summary Service."""
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the service and coordinator."""
        try:
            self.coordinator = CoordinatorAgent()
            
            log_info("Stock Summary Service initialized successfully")
            
        except Exception as e:
            log_error(f"Failed to initialize Stock Summary Service: {str(e)}")
            raise
    
    async def analyze_stock(self, query: str) -> Dict[str, Any]:
        """Route stock analysis request to coordinator and format response."""
        try:
            log_info(f"Processing stock query: {query}")
            
            # Route query to coordinator
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
        """Format successful analysis response for API."""
        final_summary = result.get("final_summary")
        
        # Handle both StockSummary object and dict cases
        if final_summary:
            if hasattr(final_summary, 'company_name'):
                # It's a StockSummary Pydantic model
                analysis_data = {
                    "company_name": final_summary.company_name,
                    "ticker": final_summary.ticker,
                    "executive_summary": final_summary.executive_summary,
                    "price_analysis": final_summary.price_analysis,
                    "news_sentiment": final_summary.news_sentiment,
                    "technical_outlook": final_summary.technical_outlook,
                    "risk_assessment": final_summary.risk_assessment,
                    "confidence_level": final_summary.confidence_level,
                    "data_sources": final_summary.data_sources or []
                }
            else:
                # It's a dictionary (fallback)
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
        else:
            # No final summary available
            analysis_data = {
                "company_name": "Unknown",
                "ticker": "Unknown",
                "executive_summary": "Analysis incomplete",
                "price_analysis": "Price analysis unavailable",
                "news_sentiment": "Sentiment analysis unavailable",
                "technical_outlook": "Technical outlook unavailable",
                "risk_assessment": "Risk assessment unavailable",
                "confidence_level": 0.0,
                "data_sources": []
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
        """Format error response for API."""
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