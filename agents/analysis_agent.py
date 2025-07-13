"""
Analysis Agent - Specialized agent for technical analysis
Handles stock price analysis, trend identification, and technical indicators
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
    TaskStatus, StockData, TechnicalAnalysis, TrendDirection
)
from utils import log_info, log_error, get_env_variable

# Configure logging
logger = logging.getLogger(__name__)

# ===============================================================================
# ANALYSIS AGENT
# ===============================================================================

class AnalysisAgent:
    """Agent responsible for technical analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the analysis agent."""
        self.config = config or {}
        self.agent_type = AgentType.ANALYSIS
        self.state = AgentState()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1000),
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        self.prompt = self._create_prompt()
    
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
        try:
            company_info = stock_data.company_info
            movements = stock_data.movements
            
            analysis_input = f"""
Company: {company_info.name} ({company_info.symbol})
Current Price: ${company_info.current_price:.2f}
Market Cap: ${company_info.market_cap:,} if available

Price Movement Analysis:
- Price Change: {movements.price_change:.2f} ({movements.percentage_change:.2f}%)
- Trend: {movements.trend}
- Volatility: {movements.volatility:.4f}
- Period High: ${movements.period_high:.2f}
- Period Low: ${movements.period_low:.2f}
- Average Volume: {movements.avg_volume:,.0f}

Recent Price History:
"""
            
            # Add recent price data
            for i, price in enumerate(stock_data.price_history[-5:]):  # Last 5 days
                analysis_input += f"Day {i+1}: Open ${price.open:.2f}, Close ${price.close:.2f}, Volume {price.volume:,}\n"
            
            return analysis_input
            
        except Exception as e:
            log_error(f"Error preparing analysis input: {str(e)}")
            return "Error preparing stock data for analysis"
    
    async def _process_analysis_result(self, result, stock_data: StockData) -> Dict[str, Any]:
        """Process analysis result and create structured output."""
        try:
            analysis_content = result.content
            
            # Extract key insights from the analysis
            output_data = {
                "technical_analysis": {
                    "trend_direction": self._extract_trend_direction(analysis_content),
                    "trend_strength": self._extract_trend_strength(analysis_content),
                    "volatility_level": self._extract_volatility_level(analysis_content),
                    "support_level": self._extract_support_level(analysis_content),
                    "resistance_level": self._extract_resistance_level(analysis_content),
                    "momentum_indicator": self._extract_momentum_indicator(analysis_content),
                    "key_insights": self._extract_key_insights(analysis_content),
                    "confidence_level": self._extract_confidence_level(analysis_content)
                },
                "analysis_summary": analysis_content,
                "price_analysis": self._generate_price_analysis_summary(analysis_content, stock_data)
            }
            
            return output_data
            
        except Exception as e:
            log_error(f"Error processing analysis result: {str(e)}")
            return {"error": str(e), "analysis_summary": str(result)}
    
    def _extract_trend_direction(self, analysis: str) -> str:
        """Extract trend direction from analysis."""
        analysis_lower = analysis.lower()
        
        if "strong upward" in analysis_lower or "strongly bullish" in analysis_lower:
            return TrendDirection.STRONG_UPWARD
        elif "upward" in analysis_lower or "bullish" in analysis_lower:
            return TrendDirection.UPWARD
        elif "strong downward" in analysis_lower or "strongly bearish" in analysis_lower:
            return TrendDirection.STRONG_DOWNWARD
        elif "downward" in analysis_lower or "bearish" in analysis_lower:
            return TrendDirection.DOWNWARD
        else:
            return TrendDirection.SIDEWAYS
    
    def _extract_trend_strength(self, analysis: str) -> float:
        """Extract trend strength from analysis."""
        analysis_lower = analysis.lower()
        
        if "very strong" in analysis_lower or "extremely strong" in analysis_lower:
            return 0.9
        elif "strong" in analysis_lower:
            return 0.8
        elif "moderate" in analysis_lower:
            return 0.6
        elif "weak" in analysis_lower:
            return 0.4
        else:
            return 0.5  # Default moderate strength
    
    def _extract_volatility_level(self, analysis: str) -> str:
        """Extract volatility level from analysis."""
        analysis_lower = analysis.lower()
        
        if "high volatility" in analysis_lower or "very volatile" in analysis_lower:
            return "High"
        elif "low volatility" in analysis_lower or "stable" in analysis_lower:
            return "Low"
        else:
            return "Medium"
    
    def _extract_support_level(self, analysis: str) -> Optional[float]:
        """Extract support level from analysis."""
        # Simple pattern matching for support levels
        import re
        
        # Look for patterns like "support at $150", "support level of $150.50"
        pattern = r'support.*?[\$]?(\d+\.?\d*)'
        match = re.search(pattern, analysis.lower())
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_resistance_level(self, analysis: str) -> Optional[float]:
        """Extract resistance level from analysis."""
        # Simple pattern matching for resistance levels
        import re
        
        # Look for patterns like "resistance at $160", "resistance level of $160.50"
        pattern = r'resistance.*?[\$]?(\d+\.?\d*)'
        match = re.search(pattern, analysis.lower())
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_momentum_indicator(self, analysis: str) -> str:
        """Extract momentum indicator from analysis."""
        analysis_lower = analysis.lower()
        
        if "bullish momentum" in analysis_lower or "positive momentum" in analysis_lower:
            return "Bullish"
        elif "bearish momentum" in analysis_lower or "negative momentum" in analysis_lower:
            return "Bearish"
        else:
            return "Neutral"
    
    def _extract_key_insights(self, analysis: str) -> List[str]:
        """Extract key insights from analysis."""
        # Simple extraction of key points
        insights = []
        
        # Split by common separators and look for actionable insights
        sentences = analysis.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in 
                ['trend', 'support', 'resistance', 'volume', 'momentum', 'outlook', 'recommend', 'suggest']):
                insights.append(sentence)
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_confidence_level(self, analysis: str) -> float:
        """Extract confidence level from analysis."""
        analysis_lower = analysis.lower()
        
        if "high confidence" in analysis_lower or "very confident" in analysis_lower:
            return 0.9
        elif "confident" in analysis_lower:
            return 0.8
        elif "moderate confidence" in analysis_lower:
            return 0.7
        elif "low confidence" in analysis_lower or "uncertain" in analysis_lower:
            return 0.5
        else:
            return 0.7  # Default moderate confidence
    
    def _generate_price_analysis_summary(self, analysis: str, stock_data: StockData) -> str:
        """Generate a concise price analysis summary."""
        try:
            company_info = stock_data.company_info
            movements = stock_data.movements
            
            summary = f"""
{company_info.name} ({company_info.symbol}) is currently trading at ${company_info.current_price:.2f}, 
showing a {movements.percentage_change:.2f}% {'gain' if movements.percentage_change > 0 else 'loss'} 
with a {movements.trend.lower()} trend. 

Key technical levels: High ${movements.period_high:.2f}, Low ${movements.period_low:.2f}. 
Volatility is at {movements.volatility:.2%} with average volume of {movements.avg_volume:,.0f} shares.

{analysis[:200]}...
"""
            
            return summary.strip()
            
        except Exception as e:
            log_error(f"Error generating price analysis summary: {str(e)}")
            return "Price analysis summary unavailable"
    
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
    async def analyze_stock(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze stock data directly."""
        try:
            # Create a task for the analysis
            task = AgentTask(
                agent_type=self.agent_type,
                description=f"Analyze stock data for {stock_data.company_info.name}",
                input_data={"stock_data": stock_data}
            )
            
            # Create a minimal workflow state
            workflow_state = WorkflowState(
                session_id="analysis_session",
                input_query=f"Analyze {stock_data.company_info.name}",
                stock_data=stock_data
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
