"""
Sentiment Agent - Specialized agent for news sentiment analysis
Handles news analysis, sentiment scoring, and theme extraction
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from models import (
    AgentType, AgentTask, AgentState, WorkflowState, AgentResponse, 
    TaskStatus, NewsData, SentimentAnalysis, SentimentType
)
from utils import log_info, log_error, get_env_variable

# Configure logging
logger = logging.getLogger(__name__)

# ===============================================================================
# SENTIMENT AGENT
# ===============================================================================

class SentimentAgent:
    """Agent responsible for news sentiment analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the sentiment agent."""
        self.config = config or {}
        self.agent_type = AgentType.SENTIMENT
        self.state = AgentState()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=self.config.get("temperature", 0.5),
            max_tokens=self.config.get("max_tokens", 1000),
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        self.prompt = self._create_prompt()
    
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
            if not news_data:
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
                    HumanMessage(content=f"Analyze the sentiment of the following news data:\n{sentiment_input}")
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
        try:
            sentiment_input = f"""
Company: {news_data.company_name} ({news_data.ticker})
Total Articles: {news_data.total_articles}
Fetch Date: {news_data.fetch_date}

Recent News Articles:
"""
            
            # Add article details
            for i, article in enumerate(news_data.articles[:10]):  # Limit to 10 articles
                sentiment_input += f"""
Article {i+1}:
- Title: {article.title}
- Source: {article.source}
- Date: {article.published_date}
- Summary: {article.summary or 'No summary available'}
- URL: {article.url}
"""
            
            return sentiment_input
            
        except Exception as e:
            log_error(f"Error preparing sentiment input: {str(e)}")
            return "Error preparing news data for sentiment analysis"
    
    async def _process_sentiment_result(self, result, news_data: NewsData) -> Dict[str, Any]:
        """Process sentiment analysis result and create structured output."""
        try:
            sentiment_content = result.content
            
            # Extract sentiment information
            output_data = {
                "sentiment_analysis": {
                    "overall_sentiment": self._extract_overall_sentiment(sentiment_content),
                    "sentiment_score": self._extract_sentiment_score(sentiment_content),
                    "positive_articles": self._count_positive_articles(sentiment_content, news_data),
                    "negative_articles": self._count_negative_articles(sentiment_content, news_data),
                    "neutral_articles": self._count_neutral_articles(sentiment_content, news_data),
                    "key_themes": self._extract_key_themes(sentiment_content),
                    "sentiment_breakdown": self._create_sentiment_breakdown(sentiment_content, news_data),
                    "confidence_level": self._extract_confidence_level(sentiment_content)
                },
                "sentiment_summary": sentiment_content,
                "news_sentiment": self._generate_news_sentiment_summary(sentiment_content, news_data)
            }
            
            return output_data
            
        except Exception as e:
            log_error(f"Error processing sentiment result: {str(e)}")
            return {"error": str(e), "sentiment_summary": str(result)}
    
    def _extract_overall_sentiment(self, sentiment: str) -> str:
        """Extract overall sentiment from analysis."""
        sentiment_lower = sentiment.lower()
        
        if "very positive" in sentiment_lower or "extremely positive" in sentiment_lower:
            return SentimentType.VERY_POSITIVE
        elif "positive" in sentiment_lower or "bullish" in sentiment_lower:
            return SentimentType.POSITIVE
        elif "very negative" in sentiment_lower or "extremely negative" in sentiment_lower:
            return SentimentType.VERY_NEGATIVE
        elif "negative" in sentiment_lower or "bearish" in sentiment_lower:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def _extract_sentiment_score(self, sentiment: str) -> float:
        """Extract sentiment score from analysis."""
        sentiment_lower = sentiment.lower()
        
        # Look for explicit scores
        import re
        score_pattern = r'score.*?(-?\d+\.?\d*)'
        match = re.search(score_pattern, sentiment_lower)
        
        if match:
            try:
                score = float(match.group(1))
                return max(-1.0, min(1.0, score))  # Clamp between -1 and 1
            except ValueError:
                pass
        
        # Fallback to keyword-based scoring
        if "very positive" in sentiment_lower or "extremely positive" in sentiment_lower:
            return 0.9
        elif "positive" in sentiment_lower:
            return 0.6
        elif "very negative" in sentiment_lower or "extremely negative" in sentiment_lower:
            return -0.9
        elif "negative" in sentiment_lower:
            return -0.6
        else:
            return 0.0
    
    def _count_positive_articles(self, sentiment: str, news_data: NewsData) -> int:
        """Count positive articles based on sentiment analysis."""
        # Simple heuristic based on analysis content
        sentiment_lower = sentiment.lower()
        
        if "mostly positive" in sentiment_lower or "predominantly positive" in sentiment_lower:
            return max(1, int(len(news_data.articles) * 0.7))
        elif "positive" in sentiment_lower:
            return max(1, int(len(news_data.articles) * 0.5))
        else:
            return int(len(news_data.articles) * 0.3)
    
    def _count_negative_articles(self, sentiment: str, news_data: NewsData) -> int:
        """Count negative articles based on sentiment analysis."""
        # Simple heuristic based on analysis content
        sentiment_lower = sentiment.lower()
        
        if "mostly negative" in sentiment_lower or "predominantly negative" in sentiment_lower:
            return max(1, int(len(news_data.articles) * 0.7))
        elif "negative" in sentiment_lower:
            return max(1, int(len(news_data.articles) * 0.5))
        else:
            return int(len(news_data.articles) * 0.3)
    
    def _count_neutral_articles(self, sentiment: str, news_data: NewsData) -> int:
        """Count neutral articles based on sentiment analysis."""
        total_articles = len(news_data.articles)
        positive_count = self._count_positive_articles(sentiment, news_data)
        negative_count = self._count_negative_articles(sentiment, news_data)
        
        return max(0, total_articles - positive_count - negative_count)
    
    def _extract_key_themes(self, sentiment: str) -> List[str]:
        """Extract key themes from sentiment analysis."""
        themes = []
        
        # Look for common financial themes
        theme_keywords = [
            "earnings", "revenue", "profit", "growth", "expansion", "acquisition",
            "merger", "competition", "market share", "innovation", "technology",
            "regulation", "lawsuit", "partnership", "leadership", "management"
        ]
        
        sentiment_lower = sentiment.lower()
        
        for keyword in theme_keywords:
            if keyword in sentiment_lower:
                # Extract context around the keyword
                context_start = max(0, sentiment_lower.find(keyword) - 50)
                context_end = min(len(sentiment), sentiment_lower.find(keyword) + 50)
                context = sentiment[context_start:context_end].strip()
                
                if len(context) > 10:
                    themes.append(context)
        
        return themes[:5]  # Return top 5 themes
    
    def _create_sentiment_breakdown(self, sentiment: str, news_data: NewsData) -> Dict[str, int]:
        """Create sentiment breakdown by category."""
        return {
            "Very Positive": 0,
            "Positive": self._count_positive_articles(sentiment, news_data),
            "Neutral": self._count_neutral_articles(sentiment, news_data),
            "Negative": self._count_negative_articles(sentiment, news_data),
            "Very Negative": 0
        }
    
    def _extract_confidence_level(self, sentiment: str) -> float:
        """Extract confidence level from sentiment analysis."""
        sentiment_lower = sentiment.lower()
        
        if "high confidence" in sentiment_lower or "very confident" in sentiment_lower:
            return 0.9
        elif "confident" in sentiment_lower:
            return 0.8
        elif "moderate confidence" in sentiment_lower:
            return 0.7
        elif "low confidence" in sentiment_lower or "uncertain" in sentiment_lower:
            return 0.5
        else:
            return 0.7  # Default moderate confidence
    
    def _generate_news_sentiment_summary(self, sentiment: str, news_data: NewsData) -> str:
        """Generate a concise news sentiment summary."""
        try:
            overall_sentiment = self._extract_overall_sentiment(sentiment)
            sentiment_score = self._extract_sentiment_score(sentiment)
            
            summary = f"""
Recent news sentiment for {news_data.company_name} ({news_data.ticker}) is {overall_sentiment.lower()} 
with a sentiment score of {sentiment_score:.2f} based on {news_data.total_articles} articles.

Key sentiment drivers include market reactions, company announcements, and industry developments.

{sentiment[:200]}...
"""
            
            return summary.strip()
            
        except Exception as e:
            log_error(f"Error generating news sentiment summary: {str(e)}")
            return "News sentiment summary unavailable"
    
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
    
    # Direct methods for coordinator to use
    async def analyze_sentiment(self, news_data: NewsData) -> Dict[str, Any]:
        """Analyze news sentiment directly."""
        try:
            # Create a task for the sentiment analysis
            task = AgentTask(
                agent_type=self.agent_type,
                description=f"Analyze sentiment for {news_data.company_name}",
                input_data={"news_data": news_data}
            )
            
            # Create a minimal workflow state
            workflow_state = WorkflowState(
                session_id="sentiment_session",
                input_query=f"Analyze sentiment for {news_data.company_name}",
                news_data=news_data
            )
            
            # Execute the task
            response = await self.execute_task(task, workflow_state)
            
            return {
                "success": response.success,
                "data": response.data,
                "message": response.message,
                "error": response.error
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ===============================================================================
# EXAMPLE USAGE
# ===============================================================================
