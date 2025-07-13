"""
Coordinator agent using LangGraph for workflow orchestration.
Manages the overall stock analysis workflow and agent interactions.
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import uuid
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models import (
    AgentType, AgentTask, WorkflowState, AgentResponse, TaskStatus,
    StockData, NewsData, TechnicalAnalysis, SentimentAnalysis,
    StockSummary, create_agent_task, create_workflow_state,
    CompanyInfo, StockMovement, NewsArticle, SentimentType, TrendDirection
)
from .agents import AgentFactory, BaseAgent
from utils import log_info, log_error, get_env_variable
from services.stock_data import StockDataFetcher
from services.ticker_lookup import TickerLookup
from services.news_fetcher import NewsFetcher

# Configure logging
logger = logging.getLogger(__name__)

# ===============================================================================
# LANGGRAPH STATE DEFINITION
# ===============================================================================

class GraphState(TypedDict):
    """State definition for LangGraph workflow."""
    # Input
    user_query: str
    session_id: str
    
    # Workflow state
    workflow_state: WorkflowState
    
    # Agent responses
    research_response: Optional[AgentResponse]
    analysis_response: Optional[AgentResponse]
    sentiment_response: Optional[AgentResponse]
    summary_response: Optional[AgentResponse]
    
    # Final output
    final_summary: Optional[StockSummary]
    
    # Execution metadata
    current_step: str
    errors: List[str]
    step_count: int
    
    # Messages for LLM context
    messages: List[BaseMessage]

# ===============================================================================
# COORDINATOR AGENT
# ===============================================================================

class CoordinatorAgent:
    """Main coordinator agent that orchestrates the entire workflow."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,  # Lower temperature for more consistent coordination
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        # Initialize specialized agents
        self.research_agent = AgentFactory.create_agent(AgentType.RESEARCH, config)
        self.analysis_agent = AgentFactory.create_agent(AgentType.ANALYSIS, config)
        self.sentiment_agent = AgentFactory.create_agent(AgentType.SENTIMENT, config)
        self.summarization_agent = AgentFactory.create_agent(AgentType.SUMMARIZATION, config)
        
        # Initialize direct data fetchers for fallback
        self.stock_fetcher = StockDataFetcher()
        self.ticker_lookup = TickerLookup()
        self.news_fetcher = NewsFetcher()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Memory saver for state persistence
        self.memory = MemorySaver()
        
        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the workflow graph
        workflow = StateGraph(GraphState)
        
        # Add nodes (workflow steps)
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("research", self._research_step)
        workflow.add_node("analysis", self._analysis_step)
        workflow.add_node("sentiment", self._sentiment_step)
        workflow.add_node("summarization", self._summarization_step)
        workflow.add_node("finalize", self._finalize_workflow)
        
        # Define the workflow edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "research")
        workflow.add_edge("research", "analysis")
        workflow.add_edge("analysis", "sentiment")
        workflow.add_edge("sentiment", "summarization")
        workflow.add_edge("summarization", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    async def process_query(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user query through the complete workflow."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Initialize the graph state
        initial_state = {
            "user_query": user_query,
            "session_id": session_id,
            "workflow_state": create_workflow_state(session_id, user_query),
            "research_response": None,
            "analysis_response": None,
            "sentiment_response": None,
            "summary_response": None,
            "final_summary": None,
            "current_step": "initialize",
            "errors": [],
            "step_count": 0,
            "messages": [HumanMessage(content=user_query)]
        }
        
        try:
            # Execute the workflow
            config = {"configurable": {"thread_id": session_id}}
            result = await self.app.ainvoke(initial_state, config=config)
            
            # Return the final result
            return {
                "success": True,
                "session_id": session_id,
                "final_summary": result.get("final_summary"),
                "workflow_state": result.get("workflow_state"),
                "step_count": result.get("step_count", 0),
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            log_error(error_msg)
            
            return {
                "success": False,
                "session_id": session_id,
                "error": error_msg,
                "errors": [error_msg]
            }
      
    async def _initialize_workflow(self, state: GraphState) -> GraphState:
        """Initialize the workflow and resolve company/ticker information."""
        log_info(f"Initializing workflow for query: {state['user_query']}")
        
        state["current_step"] = "initialize"
        state["step_count"] += 1
        
        # Extract company/ticker information from the query
        company_info = await self._extract_company_info(state["user_query"])
        
        if company_info:
            state["workflow_state"].ticker = company_info.get("ticker")
            state["workflow_state"].company_name = company_info.get("company_name")
            state["workflow_state"].update_timestamp()
            
            log_info(f"Resolved: {company_info.get('company_name')} ({company_info.get('ticker')})")
        else:
            error_msg = "Could not resolve company/ticker from query"
            state["errors"].append(error_msg)
            log_error(error_msg)
        
        return state
    
    async def _research_step(self, state: GraphState) -> GraphState:
        """Execute research phase to gather stock and news data."""
        log_info("Executing research step")
        
        state["current_step"] = "research"
        state["step_count"] += 1
        
        if not state["workflow_state"].ticker:
            error_msg = "No ticker available for research"
            state["errors"].append(error_msg)
            return state
        
        try:
            # Fetch stock data directly (more reliable than through agent)
            stock_data = await self._fetch_stock_data(
                state["workflow_state"].ticker,
                state["workflow_state"].company_name
            )
            
            if stock_data:
                state["workflow_state"].stock_data = stock_data
                log_info(f"Stock data fetched for {state['workflow_state'].ticker}")
            
            # Fetch news data
            news_data = await self._fetch_news_data(
                state["workflow_state"].company_name,
                state["workflow_state"].ticker
            )
            
            if news_data:
                state["workflow_state"].news_data = news_data
                log_info(f"News data fetched: {len(news_data.articles)} articles")
            
            # Create successful research response
            state["research_response"] = AgentResponse(
                agent_type=AgentType.RESEARCH,
                success=True,
                message="Research completed successfully",
                data={
                    "stock_data_available": stock_data is not None,
                    "news_data_available": news_data is not None
                }
            )
            
        except Exception as e:
            error_msg = f"Research step failed: {str(e)}"
            state["errors"].append(error_msg)
            log_error(error_msg)
            
            state["research_response"] = AgentResponse(
                agent_type=AgentType.RESEARCH,
                success=False,
                message=error_msg,
                error=str(e)
            )
        
        return state
    
    async def _analysis_step(self, state: GraphState) -> GraphState:
        """Execute technical analysis step."""
        log_info("Executing analysis step")
        
        state["current_step"] = "analysis"
        state["step_count"] += 1
        
        if not state["workflow_state"].stock_data:
            error_msg = "No stock data available for analysis"
            state["errors"].append(error_msg)
            return state
        
        try:
            # Create analysis task
            task = create_agent_task(
                AgentType.ANALYSIS,
                "Perform technical analysis on stock data",
                {"stock_data": state["workflow_state"].stock_data}
            )
            
            # Execute analysis
            response = await self.analysis_agent.execute_task(task, state["workflow_state"])
            
            if response.success and response.data:
                # Extract technical analysis from response
                tech_analysis = response.data.get("technical_analysis")
                if tech_analysis:
                    state["workflow_state"].technical_analysis = tech_analysis
                    log_info("Technical analysis completed")
            
            state["analysis_response"] = response
            
        except Exception as e:
            error_msg = f"Analysis step failed: {str(e)}"
            state["errors"].append(error_msg)
            log_error(error_msg)
            
            state["analysis_response"] = AgentResponse(
                agent_type=AgentType.ANALYSIS,
                success=False,
                message=error_msg,
                error=str(e)
            )
        
        return state
    
    async def _sentiment_step(self, state: GraphState) -> GraphState:
        """Execute sentiment analysis step."""
        log_info("Executing sentiment step")
        
        state["current_step"] = "sentiment"
        state["step_count"] += 1
        
        if not state["workflow_state"].news_data:
            error_msg = "No news data available for sentiment analysis"
            state["errors"].append(error_msg)
            return state
        
        try:
            # Create sentiment task
            task = create_agent_task(
                AgentType.SENTIMENT,
                "Analyze sentiment of news articles",
                {"news_data": state["workflow_state"].news_data}
            )
            
            # Execute sentiment analysis
            response = await self.sentiment_agent.execute_task(task, state["workflow_state"])
            
            if response.success and response.data:
                # Extract sentiment analysis from response
                sentiment_analysis = response.data.get("sentiment_analysis")
                if sentiment_analysis:
                    state["workflow_state"].sentiment_analysis = sentiment_analysis
                    log_info("Sentiment analysis completed")
            
            state["sentiment_response"] = response
            
        except Exception as e:
            error_msg = f"Sentiment step failed: {str(e)}"
            state["errors"].append(error_msg)
            log_error(error_msg)
            
            state["sentiment_response"] = AgentResponse(
                agent_type=AgentType.SENTIMENT,
                success=False,
                message=error_msg,
                error=str(e)
            )
        
        return state
    
    async def _summarization_step(self, state: GraphState) -> GraphState:
        """Execute summarization step."""
        log_info("Executing summarization step")
        
        state["current_step"] = "summarization"
        state["step_count"] += 1
        
        try:
            # Create summarization task
            task = create_agent_task(
                AgentType.SUMMARIZATION,
                "Create comprehensive stock summary",
                {"workflow_state": state["workflow_state"]}
            )
            
            # Execute summarization
            response = await self.summarization_agent.execute_task(task, state["workflow_state"])
            
            if response.success and response.data:
                # Extract stock summary from response
                stock_summary = response.data.get("stock_summary")
                if stock_summary:
                    state["final_summary"] = stock_summary
                    log_info("Summarization completed")
            
            state["summary_response"] = response
            
        except Exception as e:
            error_msg = f"Summarization step failed: {str(e)}"
            state["errors"].append(error_msg)
            log_error(error_msg)
            
            state["summary_response"] = AgentResponse(
                agent_type=AgentType.SUMMARIZATION,
                success=False,
                message=error_msg,
                error=str(e)
            )
        
        return state
    
    async def _finalize_workflow(self, state: GraphState) -> GraphState:
        """Finalize the workflow and prepare final output."""
        log_info("Finalizing workflow")
        
        state["current_step"] = "finalize"
        state["step_count"] += 1
        
        # Update workflow state
        state["workflow_state"].update_timestamp()
        
        # Log completion
        success_count = sum([
            1 for response in [
                state.get("research_response"),
                state.get("analysis_response"),
                state.get("sentiment_response"),
                state.get("summary_response")
            ] 
            if response and response.success
        ])
        
        log_info(f"Workflow completed: {success_count}/4 steps successful")
        
        if state["errors"]:
            log_error(f"Workflow errors: {'; '.join(state['errors'])}")
        
        return state
    
   
    async def _extract_company_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract company/ticker information from user query."""
        try:
            # Use LLM to extract company information
            prompt = f"""
            Extract the company name or stock ticker from this query: "{query}"
            
            Return the information in this format:
            Company: [company name]
            Ticker: [ticker symbol if mentioned]
            
            If no clear company is mentioned, return "None" for both.
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            # Simple parsing (could be enhanced with more sophisticated NLP)
            if "None" in response_text:
                return None
            
            # Try to lookup ticker
            company_name = None
            ticker = None
            
            # Extract potential company name from response
            if "Company:" in response_text:
                company_name = response_text.split("Company:")[1].split("\n")[0].strip()
            
            # Extract potential ticker from response
            if "Ticker:" in response_text:
                ticker = response_text.split("Ticker:")[1].split("\n")[0].strip()
            
            # If we have a company name, try to lookup ticker
            if company_name and not ticker:
                ticker = self.ticker_lookup.lookup_ticker(company_name)
            
            # If we have a ticker, verify it and get company name
            if ticker:
                company_name = self.ticker_lookup.get_company_name(ticker) or company_name
            
            if ticker and company_name:
                return {
                    "company_name": company_name,
                    "ticker": ticker
                }
            
            return None
            
        except Exception as e:
            log_error(f"Error extracting company info: {str(e)}")
            return None
    
    async def _fetch_stock_data(self, ticker: str, company_name: str) -> Optional[StockData]:
        """Fetch stock data and convert to StockData model."""
        try:
            data = self.stock_fetcher.get_comprehensive_data(ticker)
            
            if not data:
                return None
            
            # Convert to StockData model
            company_info = CompanyInfo(
                symbol=data.get("symbol", ticker),
                name=data.get("name", company_name),
                sector=data.get("sector"),
                industry=data.get("industry"),
                market_cap=data.get("market_cap"),
                current_price=data.get("current_price"),
                currency=data.get("currency", "USD"),
                exchange=data.get("exchange"),
                website=data.get("website"),
                business_summary=data.get("business_summary")
            )
            
            stock_movement = None
            if "movements" in data:
                movements = data["movements"]
                stock_movement = StockMovement(
                    first_price=movements.get("first_price", 0),
                    last_price=movements.get("last_price", 0),
                    price_change=movements.get("price_change", 0),
                    percentage_change=movements.get("percentage_change", 0),
                    trend=TrendDirection(movements.get("trend", "Sideways")),
                    volatility=movements.get("volatility", 0),
                    period_high=movements.get("period_high", 0),
                    period_low=movements.get("period_low", 0),
                    avg_volume=movements.get("avg_volume", 0)
                )
            
            return StockData(
                company_info=company_info,
                movements=stock_movement,
                data_period_days=data.get("data_period_days", 7)
            )
            
        except Exception as e:
            log_error(f"Error fetching stock data: {str(e)}")
            return None
    
    async def _fetch_news_data(self, company_name: str, ticker: str) -> Optional[NewsData]:
        """Fetch news data and convert to NewsData model."""
        try:
            articles = self.news_fetcher.get_company_news(company_name, ticker, limit=5)
            
            if not articles:
                return None
            
            # Convert to NewsArticle models
            news_articles = []
            for article in articles:
                news_article = NewsArticle(
                    title=article.get("title", ""),
                    url=article.get("url", ""),
                    summary=article.get("summary"),
                    published_date=article.get("published_date", datetime.now()),
                    source=article.get("source", "Unknown"),
                    ticker=ticker
                )
                news_articles.append(news_article)
            
            return NewsData(
                articles=news_articles,
                company_name=company_name,
                ticker=ticker
            )
            
        except Exception as e:
            log_error(f"Error fetching news data: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage of the coordinator
    import asyncio
    
    async def test_coordinator():
        coordinator = CoordinatorAgent()
        
        # Test query
        result = await coordinator.process_query("Tell me about Apple stock")
        
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Final Summary: {result.get('final_summary', {}).get('executive_summary', 'No summary')}")
        else:
            print(f"Errors: {result.get('errors', [])}")
    
    # Run the test
    asyncio.run(test_coordinator()) 