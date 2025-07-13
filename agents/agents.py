"""
Agent system for Stock Summary Agent using Langchain.
Implements specialized agents for different aspects of stock analysis.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import logging
import uuid

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field

from models import (
    AgentType, AgentTask, AgentState, WorkflowState, AgentResponse, 
    TaskStatus, StockData, NewsData, TechnicalAnalysis, SentimentAnalysis,
    CompanyInfo, StockMovement, NewsArticle, SentimentType, TrendDirection
)
from utils import log_info, log_error, get_env_variable
from services.stock_data import StockDataFetcher
from services.ticker_lookup import TickerLookup
from services.news_fetcher import NewsFetcher

# Configure logging
logger = logging.getLogger(__name__)

# ===============================================================================
# BASE AGENT CLASS
# ===============================================================================

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.config = config or {}
        self.state = AgentState()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1000),
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        
        # Create agent executor
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
    
    @abstractmethod
    def _create_tools(self) -> List[BaseTool]:
        """Create tools specific to this agent."""
        pass
    
    @abstractmethod
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for this agent."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: AgentTask, workflow_state: WorkflowState) -> AgentResponse:
        """Execute a specific task."""
        pass
    
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
        
        self.state.current_task = task if status == TaskStatus.IN_PROGRESS else None
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        max_retries = self.config.get("retry_attempts", 3)
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                log_error(f"Attempt {attempt + 1} failed for {self.agent_type}: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

# ===============================================================================
# CUSTOM TOOLS
# ===============================================================================

class StockDataTool(BaseTool):
    """Tool for fetching stock data."""
    name: str = "stock_data_fetcher"
    description: str = "Fetch comprehensive stock data for a given ticker symbol"
    
    def __init__(self):
        super().__init__()
        self.fetcher = StockDataFetcher()
    
    def _run(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Fetch stock data."""
        try:
            data = self.fetcher.get_comprehensive_data(ticker, days)
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _arun(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Async version of stock data fetching."""
        return self._run(ticker, days)

class NewsDataTool(BaseTool):
    """Tool for fetching news data."""
    name: str = "news_data_fetcher"
    description: str = "Fetch recent news articles for a given company and ticker"
    
    def __init__(self):
        super().__init__()
        self.fetcher = NewsFetcher()
    
    def _run(self, company_name: str, ticker: str, limit: int = 5) -> Dict[str, Any]:
        """Fetch news data."""
        try:
            articles = self.fetcher.get_company_news(company_name, ticker, limit)
            return {"success": True, "articles": articles}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _arun(self, company_name: str, ticker: str, limit: int = 5) -> Dict[str, Any]:
        """Async version of news data fetching."""
        return self._run(company_name, ticker, limit)

class TickerLookupTool(BaseTool):
    """Tool for looking up ticker symbols."""
    name: str = "ticker_lookup"
    description: str = "Look up stock ticker symbol for a given company name"
    
    def __init__(self):
        super().__init__()
        self.lookup = TickerLookup()
    
    def _run(self, company_name: str) -> Dict[str, Any]:
        """Look up ticker."""
        try:
            ticker = self.lookup.lookup_ticker(company_name)
            company_info = self.lookup.get_company_name(ticker) if ticker else None
            return {
                "success": True, 
                "ticker": ticker, 
                "company_name": company_info
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _arun(self, company_name: str) -> Dict[str, Any]:
        """Async version of ticker lookup."""
        return self._run(company_name)

# ===============================================================================
# RESEARCH AGENT
# ===============================================================================

class ResearchAgent(BaseAgent):
    """Agent responsible for data collection and research."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.RESEARCH, config)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create research-specific tools."""
        return [
            StockDataTool(),
            NewsDataTool(),
            TickerLookupTool()
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
    
    async def execute_task(self, task: AgentTask, workflow_state: WorkflowState) -> AgentResponse:
        """Execute research task."""
        self.update_state(task, TaskStatus.IN_PROGRESS)
        
        try:
            input_data = task.input_data
            query = input_data.get("query", "")
            
            if not query:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    message="No query provided",
                    error="Missing query parameter"
                )
            
            # Execute research using the agent
            result = await self._execute_with_retry(
                self.agent_executor.ainvoke,
                {
                    "input": f"Research the following company/stock: {query}. "
                           f"Find the ticker symbol, fetch comprehensive stock data, "
                           f"and collect recent news articles.",
                    "chat_history": self.memory.chat_memory.messages
                }
            )
            
            # Process the result and update workflow state
            output_data = await self._process_research_result(result, workflow_state)
            
            self.update_state(task, TaskStatus.COMPLETED, output_data)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=output_data,
                message="Research completed successfully"
            )
            
        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            log_error(error_msg)
            
            self.update_state(task, TaskStatus.FAILED)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                message=error_msg,
                error=str(e)
            )
    
    async def _process_research_result(self, result: Dict[str, Any], workflow_state: WorkflowState) -> Dict[str, Any]:
        """Process research results and update workflow state."""
        output_data = {
            "raw_result": result,
            "data_collected": {}
        }
        
        # Extract structured data from the result
        # This is a simplified version - in practice, you'd parse the agent's output more carefully
        if "output" in result:
            output_text = result["output"]
            # Here you would implement more sophisticated parsing
            # For now, we'll use a simple approach
            
            # Try to extract ticker from the output
            # This is a placeholder - implement proper parsing
            pass
        
        return output_data

# ===============================================================================
# ANALYSIS AGENT
# ===============================================================================

class AnalysisAgent(BaseAgent):
    """Agent responsible for technical analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.ANALYSIS, config)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create analysis-specific tools."""
        return []  # Analysis agent primarily uses LLM reasoning
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create analysis agent prompt."""
        system_message = """You are a Technical Analysis Agent specializing in stock market analysis.

Your responsibilities:
1. Analyze stock price movements and trends
2. Identify support and resistance levels
3. Assess volatility and momentum
4. Provide technical insights and predictions
5. Generate risk assessments

When analyzing stock data, consider:
- Price trends and patterns
- Volume analysis
- Volatility metrics
- Support and resistance levels
- Technical indicators
- Market context

Provide clear, actionable insights with confidence levels. Always explain your reasoning and highlight key factors influencing your analysis.

Format your analysis in a structured way with:
- Trend direction and strength
- Key price levels
- Risk factors
- Technical outlook
- Confidence assessment"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    async def execute_task(self, task: AgentTask, workflow_state: WorkflowState) -> AgentResponse:
        """Execute analysis task."""
        self.update_state(task, TaskStatus.IN_PROGRESS)
        
        try:
            # Get stock data from workflow state
            stock_data = workflow_state.stock_data
            if not stock_data:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    message="No stock data available for analysis",
                    error="Missing stock data"
                )
            
            # Prepare analysis input
            analysis_input = self._prepare_analysis_input(stock_data)
            
            # Execute analysis using the LLM
            result = await self._execute_with_retry(
                self.llm.ainvoke,
                [
                    SystemMessage(content=self.prompt.messages[0].content),
                    HumanMessage(content=f"Analyze the following stock data:\n{analysis_input}")
                ]
            )
            
            # Process the analysis result
            output_data = await self._process_analysis_result(result, stock_data)
            
            self.update_state(task, TaskStatus.COMPLETED, output_data)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=output_data,
                message="Technical analysis completed successfully"
            )
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            log_error(error_msg)
            
            self.update_state(task, TaskStatus.FAILED)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                message=error_msg,
                error=str(e)
            )
    
    def _prepare_analysis_input(self, stock_data: StockData) -> str:
        """Prepare stock data for analysis."""
        company_info = stock_data.company_info
        movements = stock_data.movements
        
        input_text = f"""
Company: {company_info.name} ({company_info.symbol})
Sector: {company_info.sector}
Current Price: ${movements.last_price:.2f}
Price Change: {movements.percentage_change:.2f}%
Trend: {movements.trend}
Volatility: {movements.volatility:.2f}%
Period High: ${movements.period_high:.2f}
Period Low: ${movements.period_low:.2f}
Average Volume: {movements.avg_volume:,.0f}
Data Period: {stock_data.data_period_days} days
"""
        return input_text
    
    async def _process_analysis_result(self, result, stock_data: StockData) -> Dict[str, Any]:
        """Process analysis results."""
        analysis_text = result.content if hasattr(result, 'content') else str(result)
        
        # Create structured technical analysis
        # This is a simplified version - in practice, you'd parse the LLM output more carefully
        technical_analysis = TechnicalAnalysis(
            trend_direction=stock_data.movements.trend,
            trend_strength=0.7,  # Would be calculated from LLM analysis
            volatility_level="Medium",  # Would be determined from analysis
            momentum_indicator="Neutral",  # Would be extracted from analysis
            key_insights=["Analysis completed"]  # Would be extracted from LLM output
        )
        
        return {
            "technical_analysis": technical_analysis,
            "analysis_text": analysis_text,
            "confidence_level": 0.75
        }

# ===============================================================================
# SENTIMENT AGENT
# ===============================================================================

class SentimentAgent(BaseAgent):
    """Agent responsible for news sentiment analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.SENTIMENT, config)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create sentiment-specific tools."""
        return []  # Sentiment agent primarily uses LLM reasoning
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create sentiment agent prompt."""
        system_message = """You are a News Sentiment Analysis Agent specializing in financial news interpretation.

Your responsibilities:
1. Analyze sentiment of news articles about companies
2. Identify key themes and topics
3. Assess potential market impact
4. Provide sentiment scores and classifications
5. Extract actionable insights from news

When analyzing news sentiment, consider:
- Overall tone and sentiment
- Key themes and topics
- Market impact potential
- Source credibility
- Recency and relevance

Classify sentiment as:
- Very Positive (+0.8 to +1.0)
- Positive (+0.3 to +0.7)
- Neutral (-0.2 to +0.2)
- Negative (-0.7 to -0.3)
- Very Negative (-1.0 to -0.8)

Provide clear explanations for your sentiment assessments and highlight key factors influencing your analysis."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    async def execute_task(self, task: AgentTask, workflow_state: WorkflowState) -> AgentResponse:
        """Execute sentiment analysis task."""
        self.update_state(task, TaskStatus.IN_PROGRESS)
        
        try:
            # Get news data from workflow state
            news_data = workflow_state.news_data
            if not news_data or not news_data.articles:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    message="No news data available for sentiment analysis",
                    error="Missing news data"
                )
            
            # Prepare sentiment analysis input
            sentiment_input = self._prepare_sentiment_input(news_data)
            
            # Execute sentiment analysis using the LLM
            result = await self._execute_with_retry(
                self.llm.ainvoke,
                [
                    SystemMessage(content=self.prompt.messages[0].content),
                    HumanMessage(content=f"Analyze the sentiment of the following news articles:\n{sentiment_input}")
                ]
            )
            
            # Process the sentiment analysis result
            output_data = await self._process_sentiment_result(result, news_data)
            
            self.update_state(task, TaskStatus.COMPLETED, output_data)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=output_data,
                message="Sentiment analysis completed successfully"
            )
            
        except Exception as e:
            error_msg = f"Sentiment analysis failed: {str(e)}"
            log_error(error_msg)
            
            self.update_state(task, TaskStatus.FAILED)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                message=error_msg,
                error=str(e)
            )
    
    def _prepare_sentiment_input(self, news_data: NewsData) -> str:
        """Prepare news data for sentiment analysis."""
        articles_text = []
        
        for i, article in enumerate(news_data.articles[:5], 1):  # Limit to 5 articles
            article_text = f"""
Article {i}:
Title: {article.title}
Source: {article.source}
Date: {article.published_date}
Summary: {article.summary or 'No summary available'}
"""
            articles_text.append(article_text)
        
        return "\n".join(articles_text)
    
    async def _process_sentiment_result(self, result, news_data: NewsData) -> Dict[str, Any]:
        """Process sentiment analysis results."""
        sentiment_text = result.content if hasattr(result, 'content') else str(result)
        
        # Create structured sentiment analysis
        # This is a simplified version - in practice, you'd parse the LLM output more carefully
        sentiment_analysis = SentimentAnalysis(
            overall_sentiment=SentimentType.NEUTRAL,  # Would be determined from LLM analysis
            sentiment_score=0.0,  # Would be calculated from analysis
            positive_articles=0,  # Would be counted from analysis
            negative_articles=0,  # Would be counted from analysis
            neutral_articles=len(news_data.articles),  # Would be counted from analysis
            key_themes=["Market analysis", "Company performance"],  # Would be extracted from LLM
            sentiment_breakdown={"neutral": len(news_data.articles)}
        )
        
        return {
            "sentiment_analysis": sentiment_analysis,
            "sentiment_text": sentiment_text,
            "confidence_level": 0.70
        }

# ===============================================================================
# SUMMARIZATION AGENT
# ===============================================================================

class SummarizationAgent(BaseAgent):
    """Agent responsible for creating final summaries."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.SUMMARIZATION, config)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create summarization-specific tools."""
        return []  # Summarization agent primarily uses LLM reasoning
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create summarization agent prompt."""
        system_message = """You are a Financial Summarization Agent specializing in creating comprehensive, readable stock analysis summaries.

Your responsibilities:
1. Synthesize all collected data into a coherent narrative
2. Create executive summaries for different audiences
3. Highlight key insights and actionable information
4. Provide balanced, objective analysis
5. Structure information for easy consumption

When creating summaries, include:
- Executive summary (2-3 sentences)
- Price analysis and trends
- News sentiment overview
- Technical outlook
- Risk assessment
- Key takeaways

Write in a professional, accessible tone. Avoid jargon where possible, and explain technical terms when necessary. 
Always provide balanced perspectives and acknowledge limitations or uncertainties in the analysis.

Structure your response with clear sections and bullet points where appropriate."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    async def execute_task(self, task: AgentTask, workflow_state: WorkflowState) -> AgentResponse:
        """Execute summarization task."""
        self.update_state(task, TaskStatus.IN_PROGRESS)
        
        try:
            # Prepare comprehensive input from all collected data
            summary_input = self._prepare_summary_input(workflow_state)
            
            # Execute summarization using the LLM
            result = await self._execute_with_retry(
                self.llm.ainvoke,
                [
                    SystemMessage(content=self.prompt.messages[0].content),
                    HumanMessage(content=f"Create a comprehensive stock analysis summary based on the following data:\n{summary_input}")
                ]
            )
            
            # Process the summarization result
            output_data = await self._process_summary_result(result, workflow_state)
            
            self.update_state(task, TaskStatus.COMPLETED, output_data)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=output_data,
                message="Summarization completed successfully"
            )
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            log_error(error_msg)
            
            self.update_state(task, TaskStatus.FAILED)
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                message=error_msg,
                error=str(e)
            )
    
    def _prepare_summary_input(self, workflow_state: WorkflowState) -> str:
        """Prepare comprehensive input for summarization."""
        sections = []
        
        # Company and stock information
        if workflow_state.stock_data:
            company_info = workflow_state.stock_data.company_info
            movements = workflow_state.stock_data.movements
            
            sections.append(f"""
COMPANY INFORMATION:
- Company: {company_info.name} ({company_info.symbol})
- Sector: {company_info.sector}
- Current Price: ${movements.last_price:.2f}
- Price Change: {movements.percentage_change:.2f}%
- Trend: {movements.trend}
- Volatility: {movements.volatility:.2f}%
- Period High: ${movements.period_high:.2f}
- Period Low: ${movements.period_low:.2f}
""")
        
        # Technical analysis
        if workflow_state.technical_analysis:
            sections.append(f"""
TECHNICAL ANALYSIS:
- Trend Direction: {workflow_state.technical_analysis.trend_direction}
- Trend Strength: {workflow_state.technical_analysis.trend_strength:.1f}
- Volatility Level: {workflow_state.technical_analysis.volatility_level}
- Momentum: {workflow_state.technical_analysis.momentum_indicator}
- Key Insights: {', '.join(workflow_state.technical_analysis.key_insights)}
""")
        
        # Sentiment analysis
        if workflow_state.sentiment_analysis:
            sections.append(f"""
SENTIMENT ANALYSIS:
- Overall Sentiment: {workflow_state.sentiment_analysis.overall_sentiment}
- Sentiment Score: {workflow_state.sentiment_analysis.sentiment_score:.2f}
- Positive Articles: {workflow_state.sentiment_analysis.positive_articles}
- Negative Articles: {workflow_state.sentiment_analysis.negative_articles}
- Key Themes: {', '.join(workflow_state.sentiment_analysis.key_themes)}
""")
        
        # News information
        if workflow_state.news_data:
            sections.append(f"""
NEWS OVERVIEW:
- Total Articles: {workflow_state.news_data.total_articles}
- Recent Headlines: {', '.join([article.title[:50] + '...' for article in workflow_state.news_data.articles[:3]])}
""")
        
        return "\n".join(sections)
    
    async def _process_summary_result(self, result, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Process summarization results."""
        summary_text = result.content if hasattr(result, 'content') else str(result)
        
        # Create structured stock summary
        # This is a simplified version - in practice, you'd parse the LLM output more carefully
        stock_summary = {
            "company_name": workflow_state.company_name or "Unknown",
            "ticker": workflow_state.ticker or "Unknown",
            "executive_summary": summary_text[:200] + "..." if len(summary_text) > 200 else summary_text,
            "price_analysis": "Analysis based on recent price movements",
            "news_sentiment": "Sentiment analysis of recent news",
            "technical_outlook": "Technical analysis outlook",
            "risk_assessment": "Risk factors and opportunities",
            "full_summary": summary_text
        }
        
        return {
            "stock_summary": stock_summary,
            "confidence_level": 0.80
        }

# ===============================================================================
# AGENT FACTORY
# ===============================================================================

class AgentFactory:
    """Factory for creating agents."""
    
    @staticmethod
    def create_agent(agent_type: AgentType, config: Dict[str, Any] = None) -> BaseAgent:
        """Create an agent of the specified type."""
        agents = {
            AgentType.RESEARCH: ResearchAgent,
            AgentType.ANALYSIS: AnalysisAgent,
            AgentType.SENTIMENT: SentimentAgent,
            AgentType.SUMMARIZATION: SummarizationAgent
        }
        
        if agent_type not in agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agents[agent_type](config)

# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================

if __name__ == "__main__":
    # Example usage of the agent system
    import asyncio
    
    async def test_agents():
        # Create a research agent
        research_agent = AgentFactory.create_agent(AgentType.RESEARCH)
        
        # Create a sample task
        task = AgentTask(
            agent_type=AgentType.RESEARCH,
            task_id="test_123",
            description="Research Apple Inc.",
            input_data={"query": "Apple Inc."}
        )
        
        # Create workflow state
        workflow_state = WorkflowState(
            session_id="test_session",
            input_query="Tell me about Apple stock"
        )
        
        # Execute the task
        response = await research_agent.execute_task(task, workflow_state)
        
        print(f"Task completed: {response.success}")
        print(f"Message: {response.message}")
    
    # Run the test
    asyncio.run(test_agents()) 