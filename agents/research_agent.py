"""
Research Agent - Specialized agent for data collection and research
Handles stock data fetching, news collection, and data validation
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

from models import (
    AgentType, AgentTask, AgentState, WorkflowState, AgentResponse, 
    TaskStatus, StockData, NewsData, CompanyInfo, StockMovement, TrendDirection, NewsArticle
)
from utils import log_info, log_error, get_env_variable
from services.stock_data import StockDataFetcher
from services.news_fetcher import NewsFetcher

# Configure logging
logger = logging.getLogger(__name__)

class StockDataTool(BaseTool):
    """Tool for fetching stock data."""
    name: str = "stock_data_fetcher"
    description: str = "Fetch comprehensive stock data for a given ticker symbol and company name"
    fetcher: StockDataFetcher = None
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'fetcher', StockDataFetcher())
    
    def _run(self, ticker: str, company_name: str, days: int = 7) -> Dict[str, Any]:
        """Run the stock data fetcher."""
        try:
            # Use the service to fetch comprehensive data
            data = self.fetcher.get_comprehensive_data(ticker, company_name, days)
            if data:
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
                
                stock_data = StockData(
                    company_info=company_info,
                    movements=stock_movement,
                    data_period_days=data.get("data_period_days", 7)
                )
                
                return {"success": True, "data": stock_data}
            else:
                return {"success": False, "error": f"No data found for ticker {ticker}"}
        except Exception as e:
            return {"success": False, "error": f"Stock data fetch failed: {str(e)}"}
    
    async def _arun(self, ticker: str, company_name: str, days: int = 7) -> Dict[str, Any]:
        """Async version of stock data fetcher."""
        return self._run(ticker, company_name, days)

class NewsDataTool(BaseTool):
    """Tool for fetching news data."""
    name: str = "news_data_fetcher"
    description: str = "Fetch recent news articles for a given company and ticker"
    fetcher: NewsFetcher = None
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'fetcher', NewsFetcher())
    
    def _run(self, company_name: str, ticker: str, limit: int = 5) -> Dict[str, Any]:
        """Run the news data fetcher."""
        try:
            # Use the service to fetch news articles
            articles_list = self.fetcher.get_company_news(company_name, ticker, limit)
            if articles_list:
                # Convert to NewsArticle models
                news_articles = []
                for article_data in articles_list:
                    try:
                        article = NewsArticle(
                            title=article_data.get("title", ""),
                            url=article_data.get("url", ""),
                            summary=article_data.get("summary", ""),
                            published_date=article_data.get("published_date"),
                            source=article_data.get("source", "Unknown"),
                            ticker=ticker
                        )
                        news_articles.append(article)
                    except Exception as e:
                        log_error(f"Error converting article to NewsArticle model: {str(e)}")
                        continue
                
                news_data = NewsData(
                    company_name=company_name,
                    ticker=ticker,
                    articles=news_articles,
                    total_articles=len(news_articles),
                    fetch_timestamp=datetime.now()
                )
                
                return {"success": True, "data": news_data, "count": len(news_articles)}
            else:
                return {"success": False, "error": f"No news found for {company_name}"}
        except Exception as e:
            return {"success": False, "error": f"News fetch failed: {str(e)}"}
    
    async def _arun(self, company_name: str, ticker: str, limit: int = 5) -> Dict[str, Any]:
        """Async version of news data fetcher."""
        return self._run(company_name, ticker, limit)

class ResearchAgent:
    """Agent responsible for data collection and research."""
    
    def __init__(self):
        """Initialize the research agent."""
        self.agent_type = AgentType.RESEARCH
        self.state = AgentState(agent_type=self.agent_type)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        
        # Create agent executor only if tools are available
        if self.tools:
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
        else:
            self.agent_executor = None

        log_info("ResearchAgent initialized successfully")
    
    def _create_tools(self) -> List[BaseTool]:
        """Create research-specific tools."""
        return [
            StockDataTool(),
            NewsDataTool()
        ]
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create research agent prompt."""
        system_message = """You are a Financial Research Agent specializing in gathering comprehensive stock market data.

Your responsibilities:
1. Resolve company names to stock ticker symbols
2. Fetch current and historical stock price data
3. Collect recent news articles about the company
4. Ensure data quality and completeness
5. Provide structured, accurate information

Always verify ticker symbols before fetching data. If you cannot find a ticker, suggest alternatives or ask for clarification.

When fetching data, prioritize:
- Accuracy and recency
- Comprehensive coverage
- Data validation
- Error handling

Format your responses with clear structure and include confidence levels for your findings."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def _get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by its name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        max_retries = 3  # Removed config dependency
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                log_error(f"Attempt {attempt + 1} failed for {self.agent_type}: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def update_state(self, task: AgentTask, status: TaskStatus, output_data: Dict[str, Any] = None):
        """Update agent state."""
        task.status = status
        if output_data:
            task.output_data = output_data
        
        if status == TaskStatus.COMPLETED:
            task.end_time = datetime.now()
            self.state.completed_tasks.append(task)
        elif status == TaskStatus.FAILED:
            task.end_time = datetime.now()
            self.state.failed_tasks.append(task)
        
        self.state.current_task = task if status == TaskStatus.IN_PROGRESS else None
    
    async def research_company(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced research method using tools to fetch data."""
        try:
            # Extract context information
            user_query = context.get("user_query", "")
            ticker = context.get("ticker")
            company_name = context.get("company_name")
            session_id = context.get("session_id", "research_session")
            workflow_state = context.get("workflow_state")
            requested_data = context.get("requested_data", [])
            
            log_info(f"Enhanced research for {company_name} ({ticker}) with context: {requested_data}")
            
            # Create task with enhanced context
            task = AgentTask(
                agent_type=self.agent_type,
                description=f"Research company: {company_name} ({ticker})",
                input_data={
                    "query": user_query,
                    "ticker": ticker,
                    "company_name": company_name,
                    "requested_data": requested_data
                }
            )
            
            # Use existing workflow state or create enhanced one
            if workflow_state:
                research_workflow_state = workflow_state
            else:
                research_workflow_state = WorkflowState(
                    session_id=session_id,
                    input_query=user_query,
                    ticker=ticker,
                    company_name=company_name
                )
            
            # Collect research data using tools
            research_data = {}
            data_sources = []
            insights = []
            
            # Fetch stock data using StockDataTool if requested
            if "stock_data" in requested_data and ticker:
                stock_tool = self._get_tool_by_name("stock_data_fetcher")
                if stock_tool:
                    try:
                        stock_result = await stock_tool._arun(ticker, company_name, days=7)
                        if stock_result.get("success"):
                            research_data["stock_data"] = stock_result["data"]
                            data_sources.append("yfinance")
                            insights.append(f"Stock data retrieved for {ticker}")
                            log_info(f"Stock data fetched for {ticker}")
                        else:
                            log_error(f"Stock data fetch failed: {stock_result.get('error')}")
                    except Exception as e:
                        log_error(f"Error using stock data tool: {str(e)}")
            
            # Fetch news data using NewsDataTool if requested
            if "news_data" in requested_data and company_name:
                news_tool = self._get_tool_by_name("news_data_fetcher")
                if news_tool:
                    try:
                        news_result = await news_tool._arun(company_name, ticker, limit=5)
                        if news_result.get("success"):
                            research_data["news_data"] = news_result["data"]
                            data_sources.extend(["yahoo_news", "google_news"])
                            insights.append(f"Found {news_result.get('count', 0)} news articles")
                            log_info(f"News data fetched: {news_result.get('count', 0)} articles")
                        else:
                            log_error(f"News data fetch failed: {news_result.get('error')}")
                    except Exception as e:
                        log_error(f"Error using news data tool: {str(e)}")
            
            # Add LangChain agent insights if available and if we don't have data yet
            if self.agent_executor and not research_data:
                try:
                    log_info(f"Using agent executor for comprehensive analysis of {company_name} ({ticker})")
                    agent_result = await self._execute_with_retry(
                        self.agent_executor.ainvoke,
                        {
                            "input": f"Research and analyze {company_name} ({ticker}). Fetch stock data and recent news, then provide key insights about the company's current status, recent developments, and market position.",
                            "chat_history": self.memory.chat_memory.messages
                        }
                    )
                    
                    if agent_result and "output" in agent_result:
                        insights.append(f"Agent Analysis: {agent_result['output'][:200]}...")
                        research_data["agent_insights"] = agent_result["output"]
                        data_sources.append("langchain_agent")
                        
                except Exception as e:
                    log_error(f"Agent executor analysis failed: {str(e)}")
            
            elif self.agent_executor and research_data:
                # If we already have data, just get AI insights without re-fetching
                try:
                    log_info(f"Getting AI insights for existing data on {company_name} ({ticker})")
                    insight_prompt = f"Based on the available data for {company_name} ({ticker}), provide key insights about the company's current status, recent developments, and market position. Do not fetch additional data."
                    
                    agent_result = await self._execute_with_retry(
                        self.llm.ainvoke,
                        [HumanMessage(content=insight_prompt)]
                    )
                    
                    if agent_result and hasattr(agent_result, 'content'):
                        insights.append(f"AI Insights: {agent_result.content[:200]}...")
                        research_data["ai_insights"] = agent_result.content
                        
                except Exception as e:
                    log_error(f"AI insights generation failed: {str(e)}")
            
            # Update workflow state
            if research_workflow_state:
                research_workflow_state.stock_data = research_data.get("stock_data")
                research_workflow_state.news_data = research_data.get("news_data")
                research_workflow_state.data_sources.extend(data_sources)
            
            # Update agent state
            self.update_state(task, TaskStatus.COMPLETED, research_data)
            
            return {
                "success": True,
                "data": {
                    **research_data,
                    "data_sources": data_sources,
                    "insights": insights
                },
                "message": f"Enhanced research completed for {company_name}",
                "workflow_state": research_workflow_state
            }
            
        except Exception as e:
            log_error(f"Enhanced research failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Enhanced research failed for {context.get('company_name', 'unknown company')}"
            }