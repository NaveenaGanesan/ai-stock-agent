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

class SummarizationAgent:
    """Agent responsible for creating final summaries."""
    
    def __init__(self):
        """Initialize the summarization agent."""
        self.agent_type = AgentType.SUMMARIZATION
        self.state = AgentState(agent_type=self.agent_type)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.6,
            max_tokens=1500,
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        self.prompt = self._create_prompt()

        log_info("SummarizationAgent initialized successfully")
    
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
        log_info(f"SummarizationAgent: Starting summarization task for {workflow_state.company_name or 'Unknown'}")
        self.update_state(task, TaskStatus.IN_PROGRESS)
        
        try:
            # Prepare summary input from all available data
            log_info("SummarizationAgent: Preparing summary input from workflow data")
            summary_input = self._prepare_summary_input(workflow_state)
            
            # Execute summarization using the LLM
            log_info("SummarizationAgent: Executing LLM summarization")
            result = await self._execute_with_retry(
                self.llm.ainvoke,
                [
                    SystemMessage(content=self.prompt.messages[0].content),
                    HumanMessage(content=f"Create a comprehensive stock summary based on the following data:\n{summary_input}")
                ]
            )
            log_info(f"SummarizationAgent: LLM summarization completed ({len(result.content)} characters)")
            
            # Process the summarization result
            log_info("SummarizationAgent: Processing summarization result")
            output_data = await self._process_summary_result(result, workflow_state)
            log_info(f"SummarizationAgent: Summarization result processed successfully")
            
            # Log key summary insights
            if 'stock_summary' in output_data:
                stock_summary = output_data['stock_summary']
                log_info(f"SummarizationAgent: Stock Summary Created:")
                log_info(f"   • Company: {stock_summary.company_name} ({stock_summary.ticker})")
                log_info(f"   • Current Price: ${stock_summary.current_price or 'N/A'}")
                log_info(f"   • Confidence Level: {stock_summary.confidence_level:.2f}")
                log_info(f"   • Data Sources: {len(stock_summary.data_sources)} sources")
            
            self.update_state(task, TaskStatus.COMPLETED, output_data)
            
            log_info("SummarizationAgent: Summarization completed successfully")
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=output_data,
                message="Summarization completed successfully"
            )
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            log_error(f"SummarizationAgent: {error_msg}")
            
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
            log_info(f"SummarizationAgent: Preparing summary input for {workflow_state.company_name or 'Unknown'}")
            
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
                log_info(f"SummarizationAgent: Stock data available for {stock_data.company_info.name}")
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
            else:
                log_error("SummarizationAgent: No stock data available in workflow state")
            
            # Add news data if available
            if workflow_state.news_data:
                news_data = workflow_state.news_data
                log_info(f"SummarizationAgent: News data available: {news_data.total_articles} articles")
                summary_input += f"""
NEWS DATA:
- Total Articles: {news_data.total_articles}
- Recent Articles: {len(news_data.articles[:3])} samples
"""
                for i, article in enumerate(news_data.articles[:3]):
                    summary_input += f"  Article {i+1}: {article.title} ({article.source})\n"
            else:
                log_error("SummarizationAgent: No news data available in workflow state")
            
            # Add technical analysis if available
            if workflow_state.technical_analysis:
                tech_analysis = workflow_state.technical_analysis
                log_info(f"🔧 Technical analysis available with {tech_analysis.trend_direction} trend")
                summary_input += f"""
TECHNICAL ANALYSIS:
- Trend Direction: {tech_analysis.trend_direction}
- Trend Strength: {tech_analysis.trend_strength}
- Volatility Level: {tech_analysis.volatility_level}
- Support Level: ${tech_analysis.support_level} if available
- Resistance Level: ${tech_analysis.resistance_level} if available
- Momentum: {tech_analysis.momentum_indicator}
"""
            else:
                log_error("SummarizationAgent: No technical analysis available in workflow state")
            
            # Add sentiment analysis if available - FIX THE ATTRIBUTE ACCESS
            if workflow_state.sentiment_analysis:
                sentiment_analysis = workflow_state.sentiment_analysis
                log_info(f"SummarizationAgent: Sentiment analysis available")
                
                # Handle both dict and object cases for sentiment analysis
                if hasattr(sentiment_analysis, 'overall_sentiment'):
                    # It's a SentimentAnalysis object
                    overall_sentiment = sentiment_analysis.overall_sentiment
                    sentiment_score = sentiment_analysis.sentiment_score
                    positive_articles = sentiment_analysis.positive_articles
                    negative_articles = sentiment_analysis.negative_articles
                    neutral_articles = sentiment_analysis.neutral_articles
                    key_themes = sentiment_analysis.key_themes[:3] if sentiment_analysis.key_themes else []
                elif isinstance(sentiment_analysis, dict):
                    # It's a dictionary
                    overall_sentiment = sentiment_analysis.get('overall_sentiment', 'Neutral')
                    sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
                    positive_articles = sentiment_analysis.get('positive_articles', 0)
                    negative_articles = sentiment_analysis.get('negative_articles', 0)
                    neutral_articles = sentiment_analysis.get('neutral_articles', 0)
                    key_themes = sentiment_analysis.get('key_themes', [])[:3]
                else:
                    log_error(f"SummarizationAgent: Unexpected sentiment analysis type: {type(sentiment_analysis)}")
                    overall_sentiment = 'Unknown'
                    sentiment_score = 0.0
                    positive_articles = 0
                    negative_articles = 0
                    neutral_articles = 0
                    key_themes = []
                
                summary_input += f"""
SENTIMENT ANALYSIS:
- Overall Sentiment: {overall_sentiment}
- Sentiment Score: {sentiment_score:.2f}
- Positive Articles: {positive_articles}
- Negative Articles: {negative_articles}
- Neutral Articles: {neutral_articles}
- Key Themes: {', '.join(key_themes) if key_themes else 'None identified'}
"""
            else:
                log_error("SummarizationAgent: No sentiment analysis available in workflow state")
            
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
            
            log_info(f"SummarizationAgent: Summary input prepared successfully ({len(summary_input)} characters)")
            return summary_input
            
        except Exception as e:
            log_error(f"SummarizationAgent: Error preparing summary input: {str(e)}")
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