#!/usr/bin/env python3
"""
Stock Summary Agent - Unified Data Models
All Pydantic models organized in a single, comprehensive class structure
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid
import os


class StockAgentModels:
    """Unified collection of all Stock Summary Agent data models"""
    
    # ===============================================================================
    # ENUMS & CONSTANTS
    # ===============================================================================
    
    class TrendDirection(str, Enum):
        """Stock trend directions."""
        STRONG_UPWARD = "Strong Upward"
        UPWARD = "Upward" 
        SIDEWAYS = "Sideways"
        DOWNWARD = "Downward"
        STRONG_DOWNWARD = "Strong Downward"

    class SentimentType(str, Enum):
        """News sentiment types."""
        VERY_POSITIVE = "Very Positive"
        POSITIVE = "Positive"
        NEUTRAL = "Neutral"
        NEGATIVE = "Negative"
        VERY_NEGATIVE = "Very Negative"

    class AgentType(str, Enum):
        """Types of agents in the system."""
        COORDINATOR = "coordinator"
        TICKER_LOOKUP = "ticker_lookup"
        RESEARCH = "research"
        ANALYSIS = "analysis"
        SENTIMENT = "sentiment"
        SUMMARIZATION = "summarization"

    class TaskStatus(str, Enum):
        """Task execution status."""
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
    
    class RiskLevel(str, Enum):
        """Risk assessment levels."""
        LOW = "low"
        MODERATE = "moderate" 
        HIGH = "high"
        VERY_HIGH = "very_high"

    # ===============================================================================
    # BASE MODELS
    # ===============================================================================
    
    class BaseDataModel(BaseModel):
        """Base model with common fields and configuration."""
        created_at: datetime = Field(default_factory=datetime.now)
        updated_at: datetime = Field(default_factory=datetime.now)
        
        class Config:
            arbitrary_types_allowed = True
            use_enum_values = True
        
        def update_timestamp(self):
            """Update the timestamp."""
            self.updated_at = datetime.now()

    # ===============================================================================
    # STOCK DATA MODELS
    # ===============================================================================
    
    class StockPrice(BaseDataModel):
        """Individual stock price data point."""
        date: datetime
        open: float = Field(gt=0, description="Opening price")
        high: float = Field(gt=0, description="High price")
        low: float = Field(gt=0, description="Low price")
        close: float = Field(gt=0, description="Closing price")
        volume: int = Field(ge=0, description="Trading volume")

    class StockMovement(BaseDataModel):
        """Stock price movement analysis."""
        first_price: float = Field(gt=0, description="First price in period")
        last_price: float = Field(gt=0, description="Last price in period")
        price_change: float = Field(description="Absolute price change")
        percentage_change: float = Field(description="Percentage change")
        trend: 'StockAgentModels.TrendDirection' = Field(description="Overall trend direction")
        volatility: float = Field(ge=0, description="Price volatility")
        period_high: float = Field(gt=0, description="Highest price in period")
        period_low: float = Field(gt=0, description="Lowest price in period")
        avg_volume: float = Field(ge=0, description="Average trading volume")

        @validator('percentage_change')
        def validate_percentage_change(cls, v):
            if abs(v) > 100:
                raise ValueError('Percentage change seems unrealistic')
            return v

    class CompanyInfo(BaseDataModel):
        """Company information."""
        symbol: str = Field(description="Stock ticker symbol")
        name: str = Field(description="Company name")
        sector: Optional[str] = Field(None, description="Business sector")
        industry: Optional[str] = Field(None, description="Industry")
        market_cap: Optional[int] = Field(None, ge=0, description="Market capitalization")
        current_price: Optional[float] = Field(None, gt=0, description="Current stock price")
        currency: str = Field(default="USD", description="Currency")
        exchange: Optional[str] = Field(None, description="Stock exchange")
        website: Optional[str] = Field(None, description="Company website")
        business_summary: Optional[str] = Field(None, description="Business description")

    class StockData(BaseDataModel):
        """Complete stock data package."""
        company_info: 'StockAgentModels.CompanyInfo'
        price_history: List['StockAgentModels.StockPrice'] = Field(default_factory=list)
        movements: Optional['StockAgentModels.StockMovement'] = None
        data_period_days: int = Field(default=7, ge=1, description="Number of days of data")
        last_updated: datetime = Field(default_factory=datetime.now)

    # ===============================================================================
    # NEWS DATA MODELS
    # ===============================================================================
    
    class NewsArticle(BaseDataModel):
        """News article information."""
        title: str = Field(description="Article title")
        url: str = Field(description="Article URL")
        summary: Optional[str] = Field(None, description="Article summary")
        published_date: datetime = Field(description="Publication date")
        source: str = Field(description="News source")
        ticker: Optional[str] = Field(None, description="Related stock ticker")
        sentiment: Optional['StockAgentModels.SentimentType'] = Field(None, description="Article sentiment")
        sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score (-1 to 1)")

    class NewsData(BaseDataModel):
        """News data collection."""
        articles: List['StockAgentModels.NewsArticle'] = Field(default_factory=list)
        company_name: str = Field(description="Company name")
        ticker: str = Field(description="Stock ticker")
        fetch_date: datetime = Field(default_factory=datetime.now)
        total_articles: int = Field(default=0, ge=0)

        @validator('total_articles', always=True)
        def set_total_articles(cls, v, values):
            if 'articles' in values:
                return len(values['articles'])
            return v

    # ===============================================================================
    # ANALYSIS MODELS
    # ===============================================================================
    
    class TechnicalAnalysis(BaseDataModel):
        """Technical analysis results."""
        trend_direction: 'StockAgentModels.TrendDirection'
        trend_strength: float = Field(ge=0, le=1, description="Trend strength (0-1)")
        volatility_level: str = Field(description="Volatility level (Low/Medium/High)")
        support_level: Optional[float] = Field(None, gt=0, description="Support price level")
        resistance_level: Optional[float] = Field(None, gt=0, description="Resistance price level")
        momentum_indicator: str = Field(description="Momentum (Bullish/Bearish/Neutral)")
        key_insights: List[str] = Field(default_factory=list, description="Key technical insights")
        indicators: Dict[str, float] = Field(default_factory=dict, description="Technical indicators")

    class SentimentAnalysis(BaseDataModel):
        """News sentiment analysis results."""
        overall_sentiment: 'StockAgentModels.SentimentType'
        sentiment_score: float = Field(ge=-1, le=1, description="Overall sentiment score")
        positive_articles: int = Field(default=0, ge=0)
        negative_articles: int = Field(default=0, ge=0)
        neutral_articles: int = Field(default=0, ge=0)
        key_themes: List[str] = Field(default_factory=list, description="Key themes in news")
        sentiment_breakdown: Dict[str, int] = Field(default_factory=dict)
        confidence_level: float = Field(default=0.7, ge=0, le=1)

    class RiskFactor(BaseDataModel):
        """Individual risk factor."""
        factor_type: str = Field(description="Type of risk factor")
        description: str = Field(description="Risk factor description")
        severity: 'StockAgentModels.RiskLevel' = Field(description="Risk severity level")
        probability: float = Field(ge=0, le=1, description="Probability of occurrence")
        impact: str = Field(description="Potential impact description")

    class Opportunity(BaseDataModel):
        """Investment opportunity."""
        opportunity_type: str = Field(description="Type of opportunity")
        description: str = Field(description="Opportunity description")
        potential_return: Optional[float] = Field(None, description="Potential return percentage")
        timeframe: str = Field(description="Expected timeframe")
        confidence: float = Field(ge=0, le=1, description="Confidence level")

    class RiskAssessment(BaseDataModel):
        """Risk assessment results."""
        overall_risk_level: 'StockAgentModels.RiskLevel'
        risk_score: float = Field(ge=0, le=1, description="Overall risk score")
        risk_factors: List['StockAgentModels.RiskFactor'] = Field(default_factory=list)
        opportunities: List['StockAgentModels.Opportunity'] = Field(default_factory=list)
        recommendations: List[str] = Field(default_factory=list)
        risk_mitigation: List[str] = Field(default_factory=list)

    class MarketContext(BaseDataModel):
        """Market context and external factors."""
        market_trend: str = Field(description="Overall market trend")
        sector_performance: Optional[str] = Field(None, description="Sector performance")
        economic_indicators: List[str] = Field(default_factory=list)
        risk_factors: List[str] = Field(default_factory=list)
        opportunities: List[str] = Field(default_factory=list)
        market_volatility: Optional[float] = Field(None, ge=0, description="Market volatility index")

    # ===============================================================================
    # AGENT WORKFLOW MODELS
    # ===============================================================================
    
    class AgentTask(BaseDataModel):
        """Individual agent task."""
        agent_type: 'StockAgentModels.AgentType'
        task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
        description: str = Field(description="Task description")
        status: 'StockAgentModels.TaskStatus' = Field(default='pending')
        input_data: Dict[str, Any] = Field(default_factory=dict)
        output_data: Dict[str, Any] = Field(default_factory=dict)
        error_message: Optional[str] = Field(None, description="Error message if failed")
        start_time: Optional[datetime] = Field(None, description="Task start time")
        end_time: Optional[datetime] = Field(None, description="Task completion time")
        
        @property
        def duration(self) -> Optional[float]:
            """Calculate task duration in seconds."""
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None

    class AgentState(BaseDataModel):
        """Agent state management."""
        agent_type: 'StockAgentModels.AgentType'
        current_task: Optional['StockAgentModels.AgentTask'] = None
        completed_tasks: List['StockAgentModels.AgentTask'] = Field(default_factory=list)
        failed_tasks: List['StockAgentModels.AgentTask'] = Field(default_factory=list)
        memory: Dict[str, Any] = Field(default_factory=dict)
        context: Dict[str, Any] = Field(default_factory=dict)
        
        def add_to_memory(self, key: str, value: Any):
            """Add item to agent memory."""
            self.memory[key] = value
        
        def get_from_memory(self, key: str, default: Any = None) -> Any:
            """Get item from agent memory."""
            return self.memory.get(key, default)

    class WorkflowState(BaseDataModel):
        """Overall workflow state."""
        session_id: str = Field(description="Unique session identifier")
        input_query: str = Field(description="Original user query")
        ticker: Optional[str] = Field(None, description="Resolved ticker symbol")
        company_name: Optional[str] = Field(None, description="Company name")
        
        # Data collected by agents
        stock_data: Optional['StockAgentModels.StockData'] = None
        news_data: Optional['StockAgentModels.NewsData'] = None
        technical_analysis: Optional['StockAgentModels.TechnicalAnalysis'] = None
        sentiment_analysis: Optional['StockAgentModels.SentimentAnalysis'] = None
        risk_assessment: Optional['StockAgentModels.RiskAssessment'] = None
        market_context: Optional['StockAgentModels.MarketContext'] = None
        
        # Workflow management
        active_tasks: List['StockAgentModels.AgentTask'] = Field(default_factory=list)
        completed_tasks: List['StockAgentModels.AgentTask'] = Field(default_factory=list)
        failed_tasks: List['StockAgentModels.AgentTask'] = Field(default_factory=list)
        
        # Execution metadata
        workflow_status: str = Field(default="initialized")
        processing_steps: List[str] = Field(default_factory=list)
        agents_used: List[str] = Field(default_factory=list)
        data_sources: List[str] = Field(default_factory=list)

    # ===============================================================================
    # OUTPUT MODELS
    # ===============================================================================
    
    class StockSummary(BaseDataModel):
        """Final stock summary output."""
        company_name: str
        ticker: str
        current_price: Optional[float] = None
        price_change: Optional[str] = None
        trend: Optional[str] = None
        
        # Analysis sections
        executive_summary: str = Field(description="Executive summary paragraph")
        price_analysis: str = Field(description="Price movement analysis")
        news_sentiment: str = Field(description="News sentiment analysis")
        technical_outlook: str = Field(description="Technical analysis outlook")
        risk_assessment: str = Field(description="Risk factors and opportunities")
        
        # Metadata
        analysis_date: datetime = Field(default_factory=datetime.now)
        confidence_level: float = Field(default=0.7, ge=0, le=1, description="Analysis confidence")
        data_sources: List[str] = Field(default_factory=list, description="Data sources used")
        
        # Raw data references
        stock_data: Optional['StockAgentModels.StockData'] = None
        news_data: Optional['StockAgentModels.NewsData'] = None
        technical_analysis: Optional['StockAgentModels.TechnicalAnalysis'] = None
        sentiment_analysis: Optional['StockAgentModels.SentimentAnalysis'] = None
        risk_analysis: Optional['StockAgentModels.RiskAssessment'] = None

    class AgentResponse(BaseDataModel):
        """Standard agent response format."""
        agent_type: 'StockAgentModels.AgentType'
        success: bool = Field(description="Whether the operation succeeded")
        data: Optional[Dict[str, Any]] = Field(None, description="Response data")
        message: str = Field(description="Response message")
        error: Optional[str] = Field(None, description="Error message if failed")
        timestamp: datetime = Field(default_factory=datetime.now)
        processing_time: Optional[float] = Field(None, description="Processing time in seconds")

    # ===============================================================================
    # CONFIGURATION MODELS
    # ===============================================================================
    
    class AgentConfig(BaseDataModel):
        """Agent configuration."""
        agent_type: 'StockAgentModels.AgentType'
        enabled: bool = Field(default=True)
        timeout_seconds: int = Field(default=30, gt=0)
        retry_attempts: int = Field(default=3, ge=0)
        temperature: float = Field(default=0.7, ge=0, le=1)
        max_tokens: int = Field(default=1000, gt=0)
        model_name: str = Field(default="gpt-4")

    class SystemConfig(BaseDataModel):
        """System-wide configuration."""
        agents: Dict['StockAgentModels.AgentType', 'StockAgentModels.AgentConfig'] = Field(default_factory=dict)
        openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
        news_api_key: Optional[str] = Field(None, description="News API key")
        default_analysis_days: int = Field(default=7, ge=1, le=30)
        max_news_articles: int = Field(default=5, ge=1, le=20)
        enable_caching: bool = Field(default=False)
        cache_ttl_minutes: int = Field(default=15, ge=1)
        
        # FastAPI configuration
        fastapi_host: str = Field(default="0.0.0.0")
        fastapi_port: int = Field(default=8000, gt=0, le=65535)
        fastapi_workers: int = Field(default=1, gt=0)
        fastapi_reload: bool = Field(default=False)
        
        # Performance settings
        max_concurrent_requests: int = Field(default=10, gt=0)
        request_timeout_seconds: int = Field(default=60, gt=0)
        
        class Config:
            env_prefix = "STOCK_AGENT_"

        @classmethod
        def from_env(cls) -> 'StockAgentModels.SystemConfig':
            """Create configuration from environment variables."""
            return cls(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                news_api_key=os.getenv("NEWS_API_KEY"),
                fastapi_host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
                fastapi_port=int(os.getenv("FASTAPI_PORT", "8000")),
                fastapi_workers=int(os.getenv("FASTAPI_WORKERS", "1")),
                fastapi_reload=os.getenv("FASTAPI_RELOAD", "false").lower() == "true",
                default_analysis_days=int(os.getenv("STOCK_AGENT_DEFAULT_ANALYSIS_DAYS", "7")),
                max_news_articles=int(os.getenv("STOCK_AGENT_MAX_NEWS_ARTICLES", "5")),
                enable_caching=os.getenv("STOCK_AGENT_ENABLE_CACHING", "false").lower() == "true",
                cache_ttl_minutes=int(os.getenv("STOCK_AGENT_CACHE_TTL_MINUTES", "15")),
            )


# ===============================================================================
# CONVENIENCE ALIASES AND FACTORY FUNCTIONS
# ===============================================================================

# Type aliases for easier imports
TrendDirection = StockAgentModels.TrendDirection
SentimentType = StockAgentModels.SentimentType
AgentType = StockAgentModels.AgentType
TaskStatus = StockAgentModels.TaskStatus
RiskLevel = StockAgentModels.RiskLevel

# Data models
StockPrice = StockAgentModels.StockPrice
StockMovement = StockAgentModels.StockMovement
CompanyInfo = StockAgentModels.CompanyInfo
StockData = StockAgentModels.StockData
NewsArticle = StockAgentModels.NewsArticle
NewsData = StockAgentModels.NewsData

# Analysis models
TechnicalAnalysis = StockAgentModels.TechnicalAnalysis
SentimentAnalysis = StockAgentModels.SentimentAnalysis
RiskFactor = StockAgentModels.RiskFactor
Opportunity = StockAgentModels.Opportunity
RiskAssessment = StockAgentModels.RiskAssessment
MarketContext = StockAgentModels.MarketContext

# Workflow models
AgentTask = StockAgentModels.AgentTask
AgentState = StockAgentModels.AgentState
WorkflowState = StockAgentModels.WorkflowState

# Output models
StockSummary = StockAgentModels.StockSummary
AgentResponse = StockAgentModels.AgentResponse

# Configuration models
AgentConfig = StockAgentModels.AgentConfig
SystemConfig = StockAgentModels.SystemConfig


# ===============================================================================
# FACTORY FUNCTIONS
# ===============================================================================

def create_agent_task(agent_type: AgentType, description: str, input_data: Dict[str, Any] = None) -> AgentTask:
    """Create a new agent task."""
    return AgentTask(
        agent_type=agent_type,
        description=description,
        input_data=input_data or {}
    )

def create_workflow_state(session_id: str, input_query: str) -> WorkflowState:
    """Create a new workflow state."""
    return WorkflowState(
        session_id=session_id,
        input_query=input_query
    )

def create_agent_response(agent_type: AgentType, success: bool, message: str, 
                         data: Dict[str, Any] = None, error: str = None) -> AgentResponse:
    """Create a standardized agent response."""
    return AgentResponse(
        agent_type=agent_type,
        success=success,
        message=message,
        data=data,
        error=error
    )

def create_system_config_from_env() -> SystemConfig:
    """Create system configuration from environment variables."""
    return SystemConfig.from_env()


# ===============================================================================
# EXAMPLE USAGE AND VALIDATION
# ===============================================================================

if __name__ == "__main__":
    # Example usage of the unified models
    
    # Create company info
    company = CompanyInfo(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        current_price=150.00,
        market_cap=2400000000000
    )
    
    # Create stock movement
    movement = StockMovement(
        first_price=148.00,
        last_price=150.00,
        price_change=2.00,
        percentage_change=1.35,
        trend=TrendDirection.UPWARD,
        volatility=0.02,
        period_high=151.00,
        period_low=147.00,
        avg_volume=50000000
    )
    
    # Create workflow state
    state = create_workflow_state("session_123", "Tell me about Apple stock")
    
    # Create agent task
    task = create_agent_task(AgentType.RESEARCH, "Research Apple Inc. stock data")
    
    # Create system config
    config = create_system_config_from_env()
    
    print("✅ All models created successfully!")
    print(f"Company: {company.name} ({company.symbol})")
    print(f"Movement: {movement.percentage_change}% ({movement.trend})")
    print(f"Workflow: {state.session_id}")
    print(f"Task: {task.task_id} - {task.description}")
    print(f"Config: FastAPI on {config.fastapi_host}:{config.fastapi_port}") 