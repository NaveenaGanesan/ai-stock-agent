"""
Ticker Lookup Agent - AI-powered company name to ticker resolution
Uses AI directly to identify and resolve company names and ticker symbols
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from models import AgentType, AgentState, TaskStatus
from utils import log_info, log_error, get_env_variable

# Configure logging
logger = logging.getLogger(__name__)


class TickerLookupAgent:
    """AI-powered agent for company name and ticker resolution."""
    
    def __init__(self):
        """Initialize the ticker lookup agent."""
        self.agent_type = AgentType.TICKER_LOOKUP
        self.state = AgentState(agent_type=self.agent_type)
        
        # Initialize LLM with low temperature for consistent results
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=300,
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        log_info("TickerLookupAgent initialized with AI-powered resolution")
    
    async def resolve_company_ticker(self, query: str) -> Dict[str, Any]:
        """
        Use AI to directly resolve company name and ticker from user query.
        
        Args:
            query: User query containing company reference
            
        Returns:
            Dictionary with resolved ticker and company info
        """
        try:
            # Update agent state - start processing
            self.update_state("processing", {"query": query, "method": "ai_direct"})
            
            log_info(f"AI resolving ticker for query: {query}")
            
            # Create AI prompt for company and ticker identification
            system_prompt = """You are an expert stock market analyst specializing in company identification and ticker symbol resolution.

Your task is to analyze user queries and extract the exact company name and corresponding stock ticker symbol.

Rules:
1. Identify the company being referenced in the query
2. Provide the exact company name and official ticker symbol
3. Handle various input formats: full names, common names, abbreviations, existing tickers
4. Return results in JSON format only
5. If you cannot identify a company with high confidence, return an error

Output format (JSON only):
{
    "success": true,
    "company_name": "Full Official Company Name",
    "ticker": "TICKER",
    "confidence": "high|medium|low"
}

For errors:
{
    "success": false,
    "error": "Reason for failure"
}

Examples:
- "Apple stock" → {"success": true, "company_name": "Apple Inc.", "ticker": "AAPL", "confidence": "high"}
- "TSLA analysis" → {"success": true, "company_name": "Tesla Inc.", "ticker": "TSLA", "confidence": "high"}
- "Microsoft Corporation" → {"success": true, "company_name": "Microsoft Corporation", "ticker": "MSFT", "confidence": "high"}"""

            user_prompt = f"Identify the company and ticker from this query: '{query}'"
            
            # Get AI response
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse AI response
            result = self._parse_ai_response(response.content, query)
            
            # Update agent state based on result
            if result["success"]:
                self.update_state("completed", {
                    "ticker": result.get("ticker"),
                    "company_name": result.get("company_name"),
                    "confidence": result.get("confidence")
                })
            else:
                self.update_state("failed", {"error": result.get("error")})
            
            log_info(f"AI ticker resolution result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"AI ticker resolution failed: {str(e)}"
            log_error(error_msg)
            
            # Update agent state as failed
            self.update_state("failed", {"error": error_msg})
            
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
    
    def _parse_ai_response(self, ai_response: str, original_query: str) -> Dict[str, Any]:
        """Parse AI response and return structured result."""
        try:
            # Clean the response - remove any markdown formatting
            cleaned_response = ai_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON response
            result = json.loads(cleaned_response)
            
            # Validate required fields for success case
            if result.get("success", False):
                if not all(key in result for key in ["company_name", "ticker"]):
                    return {
                        "success": False,
                        "error": "AI response missing required fields",
                        "query": original_query
                    }
                
                # Add metadata
                result["method"] = "ai_direct"
                result["query"] = original_query
                
                return result
            
            else:
                # Error case
                return {
                    "success": False,
                    "error": result.get("error", "AI could not identify company"),
                    "query": original_query,
                    "method": "ai_direct"
                }
                
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid AI response format: {str(e)}",
                "query": original_query,
                "ai_response": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error parsing AI response: {str(e)}",
                "query": original_query
            }
    
    async def validate_company(self, company_input: str) -> Dict[str, Any]:
        """
        Validate if input is a valid company name or ticker using AI.
        
        Args:
            company_input: Company name or ticker to validate
            
        Returns:
            Dictionary with validation result
        """
        try:
            log_info(f"AI validating company: {company_input}")
            
            system_prompt = """You are validating whether the given input is a valid publicly traded company name or ticker symbol.

Determine if the input represents a real, publicly traded company and provide the official company name and ticker.

Output JSON format:
{
    "valid": true,
    "company_name": "Official Company Name",
    "ticker": "TICKER",
    "input_type": "company_name|ticker"
}

For invalid inputs:
{
    "valid": false,
    "error": "Reason why input is invalid"
}"""

            user_prompt = f"Validate this company/ticker: '{company_input}'"
            
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse validation response
            result = self._parse_validation_response(response.content, company_input)
            
            log_info(f"AI validation result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"AI validation failed: {str(e)}"
            log_error(error_msg)
            return {
                "valid": False,
                "error": error_msg,
                "input": company_input
            }
    
    def _parse_validation_response(self, ai_response: str, original_input: str) -> Dict[str, Any]:
        """Parse AI validation response."""
        try:
            # Clean response
            cleaned_response = ai_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            result = json.loads(cleaned_response)
            result["input"] = original_input
            return result
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid AI validation response: {str(e)}",
                "input": original_input
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error parsing validation response: {str(e)}",
                "input": original_input
            }
    
    async def suggest_companies(self, query: str, limit: int = 5) -> List[str]:
        """
        Get AI-powered company suggestions based on partial query.
        
        Args:
            query: Partial company name or description
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested company names
        """
        try:
            log_info(f"AI generating company suggestions for: {query}")
            
            system_prompt = f"""Suggest up to {limit} publicly traded companies that match or are similar to the query.

Return a JSON array of company names only. Focus on:
1. Companies with similar names
2. Companies in the same industry
3. Well-known companies that might be what the user is looking for

Output format:
["Company Name 1", "Company Name 2", "Company Name 3"]

Only return the JSON array, nothing else."""

            user_prompt = f"Suggest companies similar to: '{query}'"
            
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Parse suggestions
            suggestions = self._parse_suggestions_response(response.content)
            
            log_info(f"AI suggestions: {suggestions}")
            return suggestions[:limit]
            
        except Exception as e:
            log_error(f"Error generating AI suggestions: {str(e)}")
            return []
    
    def _parse_suggestions_response(self, ai_response: str) -> List[str]:
        """Parse AI suggestions response."""
        try:
            # Clean response
            cleaned_response = ai_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            suggestions = json.loads(cleaned_response)
            
            # Ensure it's a list of strings
            if isinstance(suggestions, list):
                return [str(item) for item in suggestions if isinstance(item, str)]
            else:
                return []
                
        except json.JSONDecodeError:
            # Fallback: try to extract company names from text
            lines = ai_response.strip().split('\n')
            suggestions = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('```', '#', '*', '-')):
                    # Remove numbering and quotes
                    clean_line = line.replace('"', '').strip()
                    if clean_line and len(clean_line) > 2:
                        suggestions.append(clean_line)
            return suggestions[:5]
        except Exception:
            return []
    
    def update_state(self, status: str, data: Dict[str, Any] = None):
        """Update agent state."""
        self.state.context["status"] = status
        if data:
            self.state.context.update(data)
        self.state.context["last_updated"] = datetime.now()
