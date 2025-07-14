"""
Ticker Lookup Agent - Specialized agent for company name to ticker resolution
Handles company name resolution, ticker validation, and company suggestions
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

from models import (
    AgentType, AgentTask, AgentState, WorkflowState, AgentResponse, 
    TaskStatus, CompanyInfo
)
from utils import log_info, log_error, get_env_variable
from services.ticker_lookup import TickerLookup

# Configure logging
logger = logging.getLogger(__name__)

class TickerLookupTool(BaseTool):
    """Tool for looking up ticker symbols."""
    name: str = "ticker_lookup"
    description: str = "Look up stock ticker symbol for a given company name"
    
    def __init__(self):
        super().__init__()
        self.ticker_lookup = TickerLookup()
    
    def _run(self, company_name: str) -> Dict[str, Any]:
        """Run the ticker lookup tool."""
        try:
            ticker = self.ticker_lookup.lookup_ticker(company_name)
            if ticker:
                company_info = self.ticker_lookup.get_company_name(ticker)
                return {
                    "success": True,
                    "ticker": ticker,
                    "company_name": company_info,
                    "confidence": "high"
                }
            else:
                suggestions = self.ticker_lookup.suggest_tickers(company_name, limit=5)
                return {
                    "success": False,
                    "error": f"No ticker found for {company_name}",
                    "suggestions": suggestions
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _arun(self, company_name: str) -> Dict[str, Any]:
        """Async version of ticker lookup."""
        return self._run(company_name)

class CompanyValidationTool(BaseTool):
    """Tool for validating company names and tickers."""
    name: str = "company_validation"
    description: str = "Validate if a company name or ticker is supported"
    
    def __init__(self):
        super().__init__()
        self.ticker_lookup = TickerLookup()
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Run the company validation tool."""
        try:
            # First try as ticker
            if len(query) <= 5 and query.isupper():
                company_name = self.ticker_lookup.get_company_name(query)
                if company_name:
                    return {
                        "success": True,
                        "ticker": query,
                        "company_name": company_name,
                        "input_type": "ticker"
                    }
            
            # Try as company name
            ticker = self.ticker_lookup.lookup_ticker(query)
            if ticker:
                company_name = self.ticker_lookup.get_company_name(ticker)
                return {
                    "success": True,
                    "ticker": ticker,
                    "company_name": company_name,
                    "input_type": "company_name"
                }
            
            return {
                "success": False,
                "error": f"No match found for {query}",
                "suggestions": self.ticker_lookup.suggest_tickers(query, limit=3)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Async version of company validation."""
        return self._run(query)

class TickerLookupAgent:
    """Agent responsible for ticker lookup and company resolution."""
    
    def __init__(self):
        """Initialize the ticker lookup agent."""
        self.agent_type = AgentType.COORDINATOR  # Using coordinator type as placeholder
        self.state = AgentState()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,  # Very low temperature for consistent ticker resolution
            max_tokens=500,
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        
        # Initialize ticker lookup service
        self.ticker_lookup = TickerLookup()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create ticker lookup specific tools."""
        return [
            TickerLookupTool(),
            CompanyValidationTool()
        ]
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create ticker lookup agent prompt."""
        system_message = """You are a Ticker Lookup Agent specializing in resolving company names to stock ticker symbols.

Your responsibilities:
1. Parse user queries to extract company names or references
2. Resolve company names to accurate ticker symbols
3. Validate ticker symbols and company names
4. Provide suggestions for ambiguous queries
5. Handle various input formats (full company names, abbreviations, common names)

When processing queries, consider:
- Common company name variations (Apple, Apple Inc., AAPL)
- Abbreviations and acronyms
- Industry context clues
- Stock exchange information

Always prioritize accuracy over speed. If you're unsure about a ticker resolution, provide multiple suggestions rather than guessing.

Output format should include:
- Resolved ticker symbol
- Full company name
- Confidence level
- Alternative suggestions if applicable

Be concise but thorough in your analysis."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    async def resolve_company_ticker(self, query: str) -> Dict[str, Any]:
        """
        Resolve company name/ticker from user query.
        
        Args:
            query: User query containing company reference
            
        Returns:
            Dictionary with resolved ticker and company info
        """
        try:
            log_info(f"Resolving ticker for query: {query}")
            
            # First, try direct lookup with the ticker lookup service
            direct_result = await self._try_direct_lookup(query)
            if direct_result["success"]:
                return direct_result
            
            # If direct lookup fails, use AI to extract and resolve
            ai_result = await self._ai_assisted_lookup(query)
            
            log_info(f"Ticker resolution result: {ai_result}")
            return ai_result
            
        except Exception as e:
            error_msg = f"Ticker resolution failed: {str(e)}"
            log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
    
    async def _try_direct_lookup(self, query: str) -> Dict[str, Any]:
        """Try direct ticker lookup without AI assistance."""
        try:
            # Clean the query
            query_clean = query.strip().lower()
            
            # Common extraction patterns
            company_indicators = [
                "tell me about", "analyze", "analysis of", "stock", "shares",
                "company", "info on", "information about", "details on"
            ]
            
            # Remove common indicators
            for indicator in company_indicators:
                if indicator in query_clean:
                    query_clean = query_clean.replace(indicator, "").strip()
            
            # Remove common suffixes
            query_clean = query_clean.replace(" stock", "").replace(" shares", "")
            
            # Try ticker lookup
            ticker = self.ticker_lookup.lookup_ticker(query_clean)
            if ticker:
                company_name = self.ticker_lookup.get_company_name(ticker)
                return {
                    "success": True,
                    "ticker": ticker,
                    "company_name": company_name,
                    "confidence": "high",
                    "method": "direct_lookup"
                }
            
            return {"success": False, "method": "direct_lookup"}
            
        except Exception as e:
            return {"success": False, "error": str(e), "method": "direct_lookup"}
    
    async def _ai_assisted_lookup(self, query: str) -> Dict[str, Any]:
        """Use AI to extract company information from query."""
        try:
            # Create a focused prompt for company extraction
            extraction_prompt = f"""
            Extract the company name or ticker symbol from this query: "{query}"
            
            The query might contain:
            - Full company names (e.g., "Apple Inc.", "Microsoft Corporation")
            - Common company names (e.g., "Apple", "Microsoft")
            - Ticker symbols (e.g., "AAPL", "MSFT")
            - Informal references (e.g., "Tesla", "Amazon")
            
            Return only the most likely company name or ticker, nothing else.
            If you can't identify a company, return "UNKNOWN".
            
            Examples:
            - "Tell me about Apple stock" → "Apple"
            - "TSLA analysis" → "TSLA"
            - "Microsoft Corporation shares" → "Microsoft"
            """
            
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert at extracting company names from stock-related queries."),
                HumanMessage(content=extraction_prompt)
            ])
            
            extracted_company = response.content.strip()
            
            if extracted_company.upper() == "UNKNOWN":
                return {
                    "success": False,
                    "error": "Could not extract company name from query",
                    "query": query,
                    "method": "ai_assisted"
                }
            
            # Now try to resolve the extracted company
            ticker = self.ticker_lookup.lookup_ticker(extracted_company)
            if ticker:
                company_name = self.ticker_lookup.get_company_name(ticker)
                return {
                    "success": True,
                    "ticker": ticker,
                    "company_name": company_name,
                    "confidence": "medium",
                    "method": "ai_assisted",
                    "extracted_term": extracted_company
                }
            else:
                # Get suggestions
                suggestions = self.ticker_lookup.suggest_tickers(extracted_company, limit=3)
                return {
                    "success": False,
                    "error": f"No ticker found for '{extracted_company}'",
                    "extracted_term": extracted_company,
                    "suggestions": suggestions,
                    "method": "ai_assisted"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "ai_assisted"
            }
    
    async def validate_ticker(self, ticker: str) -> Dict[str, Any]:
        """Validate a ticker symbol."""
        try:
            company_name = self.ticker_lookup.get_company_name(ticker)
            if company_name:
                return {
                    "valid": True,
                    "ticker": ticker,
                    "company_name": company_name
                }
            else:
                return {
                    "valid": False,
                    "error": f"Invalid ticker: {ticker}"
                }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def suggest_companies(self, query: str, limit: int = 5) -> List[str]:
        """Get company suggestions based on query."""
        try:
            return self.ticker_lookup.suggest_tickers(query, limit)
        except Exception as e:
            log_error(f"Error getting suggestions: {str(e)}")
            return []
    
    async def get_supported_companies(self) -> List[str]:
        """Get all supported companies."""
        try:
            companies = list(self.ticker_lookup.common_tickers.keys())
            return sorted([company.title() for company in companies])
        except Exception as e:
            log_error(f"Error getting supported companies: {str(e)}")
            return []
    
    def update_state(self, status: str, data: Dict[str, Any] = None):
        """Update agent state."""
        self.state.context["status"] = status
        if data:
            self.state.context.update(data)
        self.state.context["last_updated"] = datetime.now()
