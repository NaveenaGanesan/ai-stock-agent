"""
Summarization Agent - Specialized agent for creating comprehensive stock summaries
Handles final summary generation, synthesis of all analysis data, and report creation
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
    TaskStatus, StockSummary, StockData, NewsData, TechnicalAnalysis, SentimentAnalysis
)
from utils import log_info, log_error, get_env_variable

# Configure logging
logger = logging.getLogger(__name__)

# ===============================================================================
# SUMMARIZATION AGENT
# ===============================================================================

class SummarizationAgent:
    """Agent responsible for creating final summaries."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the summarization agent."""
        self.config = config or {}
        self.agent_type = AgentType.SUMMARIZATION
        self.state = AgentState()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=self.config.get("temperature", 0.6),
            max_tokens=self.config.get("max_tokens", 1500),
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        self.prompt = self._create_prompt()
    
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
            # Prepare summary input from all available data
            summary_input = self._prepare_summary_input(workflow_state)
            
            # Execute summarization using the LLM
            result = await self._execute_with_retry(
                self.llm.ainvoke,
                [
                    SystemMessage(content=self.prompt.messages[0].content),
                    HumanMessage(content=f"Create a comprehensive stock summary based on the following data:\n{summary_input}")
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
        """Prepare comprehensive input for summary generation."""
        try:
            summary_input = f"""
STOCK ANALYSIS SUMMARY REQUEST

Company: {workflow_state.company_name or 'Unknown'}
Ticker: {workflow_state.ticker or 'Unknown'}
Query: {workflow_state.input_query}

=== AVAILABLE DATA ===
"""
            
            # Add stock data if available
            if workflow_state.stock_data:
                stock_data = workflow_state.stock_data
                summary_input += f"""
STOCK DATA:
- Current Price: ${stock_data.company_info.current_price:.2f}
- Market Cap: ${stock_data.company_info.market_cap:,} if available
- Price Change: {stock_data.movements.price_change:.2f} ({stock_data.movements.percentage_change:.2f}%)
- Trend: {stock_data.movements.trend}
- Volatility: {stock_data.movements.volatility:.4f}
- Period High: ${stock_data.movements.period_high:.2f}
- Period Low: ${stock_data.movements.period_low:.2f}
- Average Volume: {stock_data.movements.avg_volume:,.0f}
"""
            
            # Add news data if available
            if workflow_state.news_data:
                news_data = workflow_state.news_data
                summary_input += f"""
NEWS DATA:
- Total Articles: {news_data.total_articles}
- Recent Articles: {len(news_data.articles[:3])} samples
"""
                for i, article in enumerate(news_data.articles[:3]):
                    summary_input += f"  Article {i+1}: {article.title} ({article.source})\n"
            
            # Add technical analysis if available
            if workflow_state.technical_analysis:
                tech_analysis = workflow_state.technical_analysis
                summary_input += f"""
TECHNICAL ANALYSIS:
- Trend Direction: {tech_analysis.trend_direction}
- Trend Strength: {tech_analysis.trend_strength}
- Volatility Level: {tech_analysis.volatility_level}
- Support Level: ${tech_analysis.support_level} if available
- Resistance Level: ${tech_analysis.resistance_level} if available
- Momentum: {tech_analysis.momentum_indicator}
"""
            
            # Add sentiment analysis if available
            if workflow_state.sentiment_analysis:
                sentiment_analysis = workflow_state.sentiment_analysis
                summary_input += f"""
SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_analysis.overall_sentiment}
- Sentiment Score: {sentiment_analysis.sentiment_score:.2f}
- Positive Articles: {sentiment_analysis.positive_articles}
- Negative Articles: {sentiment_analysis.negative_articles}
- Neutral Articles: {sentiment_analysis.neutral_articles}
- Key Themes: {', '.join(sentiment_analysis.key_themes[:3])}
"""
            
            summary_input += """
=== SUMMARY REQUIREMENTS ===
Please create a comprehensive stock summary that includes:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - Key takeaway about the stock's current status
   - Most important factor influencing the stock

2. PRICE ANALYSIS
   - Current price performance
   - Technical outlook and trends
   - Key support/resistance levels

3. NEWS SENTIMENT
   - Overall sentiment from recent news
   - Key themes and drivers
   - Impact on stock perception

4. TECHNICAL OUTLOOK
   - Trend analysis and momentum
   - Volatility assessment
   - Technical indicators summary

5. RISK ASSESSMENT
   - Key risks and challenges
   - Opportunities and catalysts
   - Overall risk level

Format the response as a professional stock analysis report."""
            
            return summary_input
            
        except Exception as e:
            log_error(f"Error preparing summary input: {str(e)}")
            return "Error preparing data for summary generation"
    
    async def _process_summary_result(self, result, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Process summarization result and create structured output."""
        try:
            summary_content = result.content
            
            # Extract different sections from the summary
            sections = self._extract_summary_sections(summary_content)
            
            # Create structured stock summary
            stock_summary = self._create_stock_summary(sections, workflow_state)
            
            output_data = {
                "stock_summary": stock_summary,
                "full_summary": summary_content,
                "sections": sections,
                "generation_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model": "gpt-4",
                    "confidence": self._extract_confidence_level(summary_content)
                }
            }
            
            return output_data
            
        except Exception as e:
            log_error(f"Error processing summary result: {str(e)}")
            return {"error": str(e), "full_summary": str(result)}
    
    def _extract_summary_sections(self, summary: str) -> Dict[str, str]:
        """Extract different sections from the summary."""
        sections = {}
        
        # Define section headers to look for
        section_headers = [
            "executive summary",
            "price analysis", 
            "news sentiment",
            "technical outlook",
            "risk assessment"
        ]
        
        summary_lower = summary.lower()
        
        for header in section_headers:
            # Find the start of this section
            start_idx = summary_lower.find(header)
            if start_idx == -1:
                continue
            
            # Find the end of this section (next header or end of text)
            end_idx = len(summary)
            for other_header in section_headers:
                if other_header != header:
                    other_start = summary_lower.find(other_header, start_idx + len(header))
                    if other_start != -1 and other_start < end_idx:
                        end_idx = other_start
            
            # Extract the section content
            section_content = summary[start_idx:end_idx].strip()
            
            # Clean up the section content
            lines = section_content.split('\n')
            cleaned_lines = []
            for line in lines[1:]:  # Skip the header line
                line = line.strip()
                if line and not line.lower().startswith(tuple(section_headers)):
                    cleaned_lines.append(line)
            
            sections[header.replace(' ', '_')] = '\n'.join(cleaned_lines)
        
        return sections
    
    def _create_stock_summary(self, sections: Dict[str, str], workflow_state: WorkflowState) -> StockSummary:
        """Create a StockSummary object from extracted sections."""
        try:
            # Get basic company info
            company_name = workflow_state.company_name or "Unknown"
            ticker = workflow_state.ticker or "Unknown"
            
            # Extract current price and trend info
            current_price = None
            price_change = None
            trend = None
            
            if workflow_state.stock_data:
                current_price = workflow_state.stock_data.company_info.current_price
                movements = workflow_state.stock_data.movements
                price_change = f"{movements.price_change:.2f} ({movements.percentage_change:.2f}%)"
                trend = movements.trend
            
            # Create the stock summary
            stock_summary = StockSummary(
                company_name=company_name,
                ticker=ticker,
                current_price=current_price,
                price_change=price_change,
                trend=trend,
                executive_summary=sections.get("executive_summary", "No executive summary available"),
                price_analysis=sections.get("price_analysis", "No price analysis available"),
                news_sentiment=sections.get("news_sentiment", "No sentiment analysis available"),
                technical_outlook=sections.get("technical_outlook", "No technical outlook available"),
                risk_assessment=sections.get("risk_assessment", "No risk assessment available"),
                confidence_level=self._extract_confidence_level(sections.get("executive_summary", "")),
                data_sources=self._get_data_sources(workflow_state),
                stock_data=workflow_state.stock_data,
                news_data=workflow_state.news_data,
                technical_analysis=workflow_state.technical_analysis,
                sentiment_analysis=workflow_state.sentiment_analysis
            )
            
            return stock_summary
            
        except Exception as e:
            log_error(f"Error creating stock summary: {str(e)}")
            # Return a minimal summary in case of error
            return StockSummary(
                company_name=workflow_state.company_name or "Unknown",
                ticker=workflow_state.ticker or "Unknown",
                executive_summary="Error generating summary",
                price_analysis="Error generating price analysis",
                news_sentiment="Error generating sentiment analysis",
                technical_outlook="Error generating technical outlook",
                risk_assessment="Error generating risk assessment"
            )
    
    def _extract_confidence_level(self, content: str) -> float:
        """Extract confidence level from summary content."""
        content_lower = content.lower()
        
        if "high confidence" in content_lower or "very confident" in content_lower:
            return 0.9
        elif "confident" in content_lower:
            return 0.8
        elif "moderate confidence" in content_lower:
            return 0.7
        elif "low confidence" in content_lower or "uncertain" in content_lower:
            return 0.5
        else:
            return 0.7  # Default moderate confidence
    
    def _get_data_sources(self, workflow_state: WorkflowState) -> List[str]:
        """Get list of data sources used in the analysis."""
        sources = []
        
        if workflow_state.stock_data:
            sources.append("Stock Data (yfinance)")
        
        if workflow_state.news_data:
            sources.append("News Data")
        
        if workflow_state.technical_analysis:
            sources.append("Technical Analysis")
        
        if workflow_state.sentiment_analysis:
            sources.append("Sentiment Analysis")
        
        return sources
    
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
    async def create_summary(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Create a comprehensive summary directly."""
        try:
            # Create a task for the summarization
            task = AgentTask(
                agent_type=self.agent_type,
                description=f"Create summary for {workflow_state.company_name}",
                input_data={"workflow_state": workflow_state}
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
