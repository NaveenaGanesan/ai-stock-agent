"""
Models package for the AI Stock Agent application.
Contains all Pydantic models and data structures.
"""

from .models import StockAgentModels
from .api_models import (
    AnalysisRequest,
    BatchAnalysisRequest,
    AnalysisResponse,
    BatchAnalysisResponse,
    HealthResponse,
)

# Convenience imports for easy access
from .models import (
    create_agent_task,
    create_workflow_state,
    create_agent_response,
    create_system_config_from_env,
)

# Create aliases for easy access to inner classes
TrendDirection = StockAgentModels.TrendDirection
SentimentType = StockAgentModels.SentimentType
AgentType = StockAgentModels.AgentType
TaskStatus = StockAgentModels.TaskStatus
RiskLevel = StockAgentModels.RiskLevel

# Base Models
BaseDataModel = StockAgentModels.BaseDataModel

# Stock Data Models
StockPrice = StockAgentModels.StockPrice
StockMovement = StockAgentModels.StockMovement
CompanyInfo = StockAgentModels.CompanyInfo
StockData = StockAgentModels.StockData

# News Models
NewsArticle = StockAgentModels.NewsArticle
NewsData = StockAgentModels.NewsData

# Analysis Models
TechnicalAnalysis = StockAgentModels.TechnicalAnalysis
SentimentAnalysis = StockAgentModels.SentimentAnalysis
RiskFactor = StockAgentModels.RiskFactor
Opportunity = StockAgentModels.Opportunity
RiskAssessment = StockAgentModels.RiskAssessment
MarketContext = StockAgentModels.MarketContext

# Workflow Models
AgentTask = StockAgentModels.AgentTask
AgentState = StockAgentModels.AgentState
WorkflowState = StockAgentModels.WorkflowState
AgentResponse = StockAgentModels.AgentResponse
AgentConfig = StockAgentModels.AgentConfig

# Output Models
StockSummary = StockAgentModels.StockSummary
SystemConfig = StockAgentModels.SystemConfig

__all__ = [
    'StockAgentModels',
    # Enums
    'TrendDirection',
    'SentimentType',
    'AgentType',
    'TaskStatus',
    'RiskLevel',
    # Base Models
    'BaseDataModel',
    # Stock Data Models
    'StockPrice',
    'StockMovement',
    'CompanyInfo',
    'StockData',
    # News Models
    'NewsArticle',
    'NewsData',
    # Analysis Models
    'TechnicalAnalysis',
    'SentimentAnalysis',
    'RiskFactor',
    'Opportunity',
    'RiskAssessment',
    'MarketContext',
    # Workflow Models
    'AgentTask',
    'AgentState',
    'WorkflowState',
    'AgentResponse',
    'AgentConfig',
    # Output Models
    'StockSummary',
    'SystemConfig',
    # API Models
    'AnalysisRequest',
    'BatchAnalysisRequest',
    'AnalysisResponse',
    'BatchAnalysisResponse',
    'HealthResponse',
    # Factory functions
    'create_agent_task',
    'create_workflow_state',
    'create_agent_response',
    'create_system_config_from_env',
] 