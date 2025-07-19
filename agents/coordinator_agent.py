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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from models import (
    AgentType, AgentTask, AgentState, WorkflowState, AgentResponse, TaskStatus,
    StockData, NewsData, TechnicalAnalysis, SentimentAnalysis,
    StockSummary, create_agent_task, create_workflow_state,
    CompanyInfo, StockMovement, NewsArticle, SentimentType, TrendDirection
)
from .ticker_lookup_agent import TickerLookupAgent
from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from .sentiment_agent import SentimentAgent
from .summarization_agent import SummarizationAgent
from utils import log_info, log_error, get_env_variable
from services.stock_data import StockDataFetcher
from services.news_fetcher import NewsFetcher

# Configure logging
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State definition for LangGraph workflow."""
    # Input
    user_query: str
    session_id: str
    
    # Workflow state
    workflow_state: WorkflowState
    
    # Agent responses
    ticker_lookup_response: Optional[AgentResponse]
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

class CoordinatorAgent:
    """Main coordinator agent that orchestrates the entire workflow."""
    
    def __init__(self):
        log_info("Initializing CoordinatorAgent")
        
        # Initialize coordinator's own agent state
        self.agent_type = AgentType.COORDINATOR
        self.state = AgentState(agent_type=self.agent_type)
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,  # Lower temperature for more consistent coordination
            openai_api_key=get_env_variable("OPENAI_API_KEY")
        )
        
        # Initialize specialized agents
        self.ticker_lookup_agent = TickerLookupAgent()
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.sentiment_agent = SentimentAgent()
        self.summarization_agent = SummarizationAgent()
        
        # Initialize direct data fetchers for fallback
        self.stock_fetcher = StockDataFetcher()
        self.news_fetcher = NewsFetcher()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Compile the graph without session persistence
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the workflow graph
        workflow = StateGraph(GraphState)
        
        # Add nodes (workflow steps)
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("ticker_lookup", self._ticker_lookup_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("analysis", self._analysis_step)
        workflow.add_node("sentiment", self._sentiment_step)
        workflow.add_node("summarization", self._summarization_step)
        workflow.add_node("finalize", self._finalize_workflow)
        
        # Define the workflow edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "ticker_lookup")
        workflow.add_edge("ticker_lookup", "research")
        workflow.add_edge("research", "analysis")
        workflow.add_edge("analysis", "sentiment")
        workflow.add_edge("sentiment", "summarization")
        workflow.add_edge("summarization", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the complete workflow."""
        # Generate a workflow ID for tracking
        workflow_id = str(uuid.uuid4())
        
        # Initialize the graph state
        initial_state = {
            "user_query": user_query,
            "session_id": workflow_id,
            "workflow_state": create_workflow_state(workflow_id, user_query),
            "ticker_lookup_response": None,
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
            result = await self.app.ainvoke(initial_state)
            
            # Return the final result
            return {
                "success": True,
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
                "error": error_msg,
                "errors": [error_msg]
            }
      
    async def _initialize_workflow(self, state: GraphState) -> GraphState:
        """Initialize the workflow."""
        log_info(f"Initializing workflow for query: {state['user_query']}")
        
        state["current_step"] = "initialize"
        state["step_count"] += 1
        
        # Basic initialization - ticker lookup will be handled in the next step
        state["workflow_state"].update_timestamp()
        
        return state
    
    async def _ticker_lookup_step(self, state: GraphState) -> GraphState:
        """Execute ticker lookup step using the dedicated agent."""
        log_info("Executing ticker lookup step")
        
        state["current_step"] = "ticker_lookup"
        state["step_count"] += 1
        
        try:
            # Create a task for coordinator's AgentState tracking
            task = AgentTask(
                agent_type=AgentType.COORDINATOR,
                description=f"Ticker lookup for query: {state['user_query']}",
                input_data={"query": state["user_query"]}
            )
            
            # Update coordinator's state
            self._update_coordinator_state(task, TaskStatus.IN_PROGRESS)
            
            # Use the ticker lookup agent to resolve company/ticker
            result = await self.ticker_lookup_agent.resolve_company_ticker(state["user_query"])
            
            if result["success"]:
                # Update WorkflowState
                state["workflow_state"].ticker = result.get("ticker")
                state["workflow_state"].company_name = result.get("company_name")
                state["workflow_state"].update_timestamp()
                
                # Create successful ticker lookup response
                state["ticker_lookup_response"] = AgentResponse(
                    agent_type=AgentType.COORDINATOR,  # Coordinator handles ticker lookup
                    success=True,
                    message=f"Resolved: {result.get('company_name')} ({result.get('ticker')})",
                    data={
                        "ticker": result.get("ticker"),
                        "company_name": result.get("company_name"),
                        "confidence": result.get("confidence", "medium"),
                        "method": result.get("method", "ai_direct")
                    }
                )
                
                # Update coordinator's state as completed
                self._update_coordinator_state(task, TaskStatus.COMPLETED, result)
                
                # Update shared memory for subsequent agents
                self._update_shared_memory(AgentType.COORDINATOR, "ticker_resolved", True)
                self._update_shared_memory(AgentType.COORDINATOR, "ticker", result.get("ticker"))
                self._update_shared_memory(AgentType.COORDINATOR, "company_name", result.get("company_name"))
                
                log_info(f"Resolved: {result.get('company_name')} ({result.get('ticker')})")
                
            else:
                error_msg = f"Could not resolve company/ticker: {result.get('error')}"
                state["errors"].append(error_msg)
                
                # Create failed ticker lookup response
                state["ticker_lookup_response"] = AgentResponse(
                    agent_type=AgentType.COORDINATOR,
                    success=False,
                    message=error_msg,
                    error=result.get("error")
                )
                
                # Update coordinator's state as failed
                self._update_coordinator_state(task, TaskStatus.FAILED)
                
                log_error(error_msg)
        
        except Exception as e:
            error_msg = f"Ticker lookup step failed: {str(e)}"
            state["errors"].append(error_msg)
            
            # Create error response
            state["ticker_lookup_response"] = AgentResponse(
                agent_type=AgentType.COORDINATOR,
                success=False,
                message=error_msg,
                error=str(e)
            )
            
            log_error(error_msg)
        
        return state
    
    def _update_coordinator_state(self, task: AgentTask, status: TaskStatus, output_data: Dict[str, Any] = None):
        """Update coordinator's AgentState."""
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
    
    def _update_shared_memory(self, agent_type: AgentType, key: str, value: Any):
        """Update shared memory across all agents."""
        try:
            # Update coordinator's context
            self.state.add_to_memory(f"{agent_type.value}_{key}", value)
            
            # Create context message for agent memories
            context_message = f"Context from {agent_type.value}: {key} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}"
            
            # Update relevant agent memories using proper LangChain API
            if agent_type == AgentType.RESEARCH:
                self.research_agent.memory.chat_memory.add_message(AIMessage(content=context_message))
            elif agent_type == AgentType.ANALYSIS:
                self.analysis_agent.memory.chat_memory.add_message(AIMessage(content=context_message))
            elif agent_type == AgentType.SENTIMENT:
                self.sentiment_agent.memory.chat_memory.add_message(AIMessage(content=context_message))
            elif agent_type == AgentType.SUMMARIZATION:
                self.summarization_agent.memory.chat_memory.add_message(AIMessage(content=context_message))
                
            log_info(f"Updated shared memory: {agent_type.value}.{key}")
            
        except Exception as e:
            log_error(f"Failed to update shared memory: {str(e)}")
    
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
            # Pass comprehensive context to research agent
            research_context = {
                "user_query": state["user_query"],
                "ticker": state["workflow_state"].ticker,
                "company_name": state["workflow_state"].company_name,
                "session_id": state["session_id"],
                "workflow_state": state["workflow_state"],
                "previous_steps": state["step_count"] - 1,
                "requested_data": ["stock_data", "news_data", "company_info"]
            }
            
            # Use enhanced research method
            result = await self.research_agent.research_company(research_context)
            
            if result["success"]:
                # Update workflow state with research results
                research_data = result["data"]
                
                if research_data.get("stock_data"):
                    state["workflow_state"].stock_data = research_data["stock_data"]
                    log_info(f"Stock data received from research agent for {state['workflow_state'].ticker}")
                
                if research_data.get("news_data"):
                    state["workflow_state"].news_data = research_data["news_data"]
                    log_info(f"News data received: {len(research_data['news_data'].articles)} articles")
                
                # Update data sources tracking
                state["workflow_state"].data_sources.extend(research_data.get("data_sources", []))
                
                # Create successful research response
                state["research_response"] = AgentResponse(
                    agent_type=AgentType.RESEARCH,
                    success=True,
                    message="Research completed successfully",
                    data={
                        "stock_data_available": research_data.get("stock_data") is not None,
                        "news_data_available": research_data.get("news_data") is not None,
                        "data_sources": research_data.get("data_sources", []),
                        "research_insights": research_data.get("insights", [])
                    }
                )
                
                # Add research insights to messages for context
                if research_data.get("insights"):
                    insights_msg = f"Research findings: {'; '.join(research_data['insights'])}"
                    state["messages"].append(AIMessage(content=insights_msg))
                
                # Update shared memory for subsequent agents
                self._update_shared_memory(AgentType.RESEARCH, "completed", True)
                self._update_shared_memory(AgentType.RESEARCH, "ticker", state["workflow_state"].ticker)
                self._update_shared_memory(AgentType.RESEARCH, "company_name", state["workflow_state"].company_name)
                self._update_shared_memory(AgentType.RESEARCH, "data_sources", research_data.get("data_sources", []))
                self._update_shared_memory(AgentType.RESEARCH, "insights", research_data.get("insights", []))
                
            else:
                error_msg = f"Research failed: {result.get('error')}"
                state["errors"].append(error_msg)
                log_error(error_msg)
                
                state["research_response"] = AgentResponse(
                    agent_type=AgentType.RESEARCH,
                    success=False,
                    message=error_msg,
                    error=result.get("error")
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
            log_error(f"CoordinatorAgent: {error_msg}")
            return state
        
        try:
            log_info(f"CoordinatorAgent: Starting technical analysis for {state['workflow_state'].company_name}")
            
            # Use the analysis agent to analyze stock data
            result = await self.analysis_agent.analyze_stock(state["workflow_state"].stock_data)
            
            if result["success"]:
                # Extract technical analysis from response and convert to proper model
                tech_analysis_data = result["data"].get("technical_analysis")
                if tech_analysis_data:
                    # Convert dictionary to TechnicalAnalysis model
                    technical_analysis = TechnicalAnalysis(
                        trend_direction=tech_analysis_data.get("trend_direction", TrendDirection.SIDEWAYS),
                        trend_strength=tech_analysis_data.get("trend_strength", 0.5),
                        volatility_level=tech_analysis_data.get("volatility_level", "Medium"),
                        support_level=tech_analysis_data.get("support_level"),
                        resistance_level=tech_analysis_data.get("resistance_level"),
                        momentum_indicator=tech_analysis_data.get("momentum_indicator", "Neutral"),
                        key_insights=tech_analysis_data.get("key_insights", []),
                        confidence_level=tech_analysis_data.get("confidence_level", 0.7)
                    )
                    
                    # Store the proper model object in workflow state
                    state["workflow_state"].technical_analysis = technical_analysis
                    log_info(f"CoordinatorAgent: Technical analysis completed with {technical_analysis.trend_direction} trend")
                
                state["analysis_response"] = AgentResponse(
                    agent_type=AgentType.ANALYSIS,
                    success=True,
                    message="Analysis completed successfully",
                    data=result["data"]
                )
            else:
                error_msg = f"Analysis failed: {result.get('error')}"
                state["errors"].append(error_msg)
                log_error(f"CoordinatorAgent: {error_msg}")
                
                state["analysis_response"] = AgentResponse(
                    agent_type=AgentType.ANALYSIS,
                    success=False,
                    message=error_msg,
                    error=result.get("error")
                )
            
        except Exception as e:
            error_msg = f"Analysis step failed: {str(e)}"
            state["errors"].append(error_msg)
            log_error(f"CoordinatorAgent: {error_msg}")
            
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
            log_error(f"CoordinatorAgent: {error_msg}")
            return state
        
        try:
            log_info(f"CoordinatorAgent: Starting sentiment analysis for {state['workflow_state'].company_name}")
            
            # Use the sentiment agent to analyze news data
            result = await self.sentiment_agent.analyze_sentiment(state["workflow_state"].news_data)
            
            if result["success"]:
                # Extract sentiment analysis from response and convert to proper model
                sentiment_analysis_data = result["data"].get("sentiment_analysis")
                if sentiment_analysis_data:
                    # Convert dictionary to SentimentAnalysis model
                    sentiment_analysis = SentimentAnalysis(
                        overall_sentiment=SentimentType(sentiment_analysis_data.get("overall_sentiment", SentimentType.NEUTRAL)),
                        sentiment_score=sentiment_analysis_data.get("sentiment_score", 0.0),
                        positive_articles=sentiment_analysis_data.get("positive_articles", 0),
                        negative_articles=sentiment_analysis_data.get("negative_articles", 0),
                        neutral_articles=sentiment_analysis_data.get("neutral_articles", 0),
                        key_themes=sentiment_analysis_data.get("key_themes", []),
                        sentiment_breakdown=sentiment_analysis_data.get("sentiment_breakdown", {}),
                        confidence_level=sentiment_analysis_data.get("confidence_level", 0.7)
                    )
                    
                    # Store the proper model object in workflow state
                    state["workflow_state"].sentiment_analysis = sentiment_analysis
                    log_info(f"CoordinatorAgent: Sentiment analysis completed - {sentiment_analysis.overall_sentiment} sentiment")
                
                state["sentiment_response"] = AgentResponse(
                    agent_type=AgentType.SENTIMENT,
                    success=True,
                    message="Sentiment analysis completed successfully",
                    data=result["data"]
                )
            else:
                error_msg = f"Sentiment analysis failed: {result.get('error')}"
                state["errors"].append(error_msg)
                log_error(f"CoordinatorAgent: {error_msg}")
                
                state["sentiment_response"] = AgentResponse(
                    agent_type=AgentType.SENTIMENT,
                    success=False,
                    message=error_msg,
                    error=result.get("error")
                )
            
        except Exception as e:
            error_msg = f"Sentiment step failed: {str(e)}"
            state["errors"].append(error_msg)
            log_error(f"CoordinatorAgent: {error_msg}")
            
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
            # Use the summarization agent to create summary
            result = await self.summarization_agent.create_summary(state["workflow_state"])
            
            if result["success"]:
                # Extract stock summary from response
                stock_summary = result["data"].get("stock_summary")
                if stock_summary:
                    state["final_summary"] = stock_summary
                    log_info("Summarization completed")
                
                state["summary_response"] = AgentResponse(
                    agent_type=AgentType.SUMMARIZATION,
                    success=True,
                    message="Summarization completed successfully",
                    data=result["data"]
                )
            else:
                error_msg = f"Summarization failed: {result.get('error')}"
                state["errors"].append(error_msg)
                log_error(error_msg)
                
                state["summary_response"] = AgentResponse(
                    agent_type=AgentType.SUMMARIZATION,
                    success=False,
                    message=error_msg,
                    error=result.get("error")
                )
            
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
