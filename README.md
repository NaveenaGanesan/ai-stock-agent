# AI Stock Agent

A sophisticated AI-powered stock analysis platform featuring a complete agent-based architecture with LangChain/LangGraph orchestration. The system provides both FastAPI REST endpoints and a command-line interface for comprehensive stock market analysis.

## 🏗️ Architecture Overview

**Complete Agent-Based System**: Every component is an intelligent agent that collaborates through LangGraph workflows:

```mermaid
graph TB
    subgraph "Client Layer"
        A[REST API Client] --> B[FastAPI Backend]
    end
    
    subgraph "Coordination & Orchestration Layer"
        B --> C[Coordinator Agent]
        C --> D[LangGraph StateGraph]
    end
    
    subgraph "Agent Execution Layer"
        D --> E[Ticker Lookup Agent]
        D --> F[Research Agent]
        D --> G[Analysis Agent]
        D --> H[Sentiment Agent]
        D --> I[Summarization Agent]
    end
    
    subgraph "Tool Layer"
        F --> J[StockDataTool]
        F --> K[NewsDataTool]
        E --> L[AI Lookup Tool]
    end
    
    subgraph "Service Layer"
        J --> M[StockDataFetcher]
        K --> N[NewsFetcher]
        L --> O[LLM Service]
    end
    
    subgraph "External Data Sources"
        M --> P[Yahoo Finance API]
        N --> Q[NewsAPI]
        N --> R[Google News]
        O --> S[OpenAI GPT-4]
    end
```

## 🔄 State Management Architecture

The AI Stock Agent uses a sophisticated multi-layered state management system powered by LangGraph:

### State Layers Explained

1. **GraphState** - Top-level workflow orchestration state
   - Manages overall execution flow and agent coordination
   - Tracks current step, errors, and execution metadata
   - Contains references to all other state layers

2. **WorkflowState** - Business logic and data aggregation state
   - Stores ticker, company name, and user query
   - Accumulates data from all agents (stock_data, news_data, etc.)
   - Tracks data sources and processing steps

3. **AgentState** - Individual agent execution state
   - Manages agent-specific tasks and their status
   - Tracks completed, failed, and active tasks
   - Maintains agent-specific memory and context

4. **Chat Conversations** - Conversational context management
   - `ConversationBufferMemory`: Stores chat history for each agent
   - **Context Retention**: Agents remember previous interactions
   - **Contextual Analysis**: Each analysis builds on conversation history
   - **Multi-turn Support**: Supports follow-up questions and refinements
   - **Memory Sharing**: Agents can access shared conversation context for continuity

### Chat-Enabled Features

- **Follow-up Questions**: "Tell me more about Apple's technical indicators"
- **Contextual Refinement**: "Focus on the last 30 days instead"
- **Cross-reference Queries**: "How does this compare to Microsoft?"
- **Progressive Analysis**: Each interaction builds deeper insights
- **Conversation History**: Full audit trail of user interactions

### State Flow Benefits

- **Fault Tolerance**: State can be recovered at any step if agents fail
- **Transparency**: Full visibility into what each agent accomplished
- **Debugging**: Easy to trace where issues occur in the workflow
- **Extensibility**: New agents can easily access existing state data
- **Conversation Continuity**: Maintains context across multiple interactions

## 📁 Project Structure

```
ai-stock-agent/
├── app.py                           # Unified FastAPI Backend + CLI Interface
├── models/                          # Data Models & Schemas
│   ├── __init__.py
│   └── models.py                   # Unified StockAgentModels class
├── agents/                          # AI Agents & Orchestration
│   ├── __init__.py
│   ├── coordinator_agent.py        # LangGraph workflow coordinator
│   ├── ticker_lookup_agent.py      # Company/ticker resolution agent
│   ├── research_agent.py           # Data collection & research agent
│   ├── analysis_agent.py           # Technical analysis agent
│   ├── sentiment_agent.py          # News sentiment analysis agent
│   └── summarization_agent.py      # Final summary generation agent
├── services/                        # External Services Integration
│   ├── __init__.py
│   ├── stock_data.py               # Stock market data fetching service
│   └── news_fetcher.py             # News article collection service
├── utils/                           # Utility Functions
│   ├── __init__.py
│   └── utils.py                    # Helper functions & formatting
├── env.example                      # Environment variables template
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🔧 LangChain & LangGraph Architecture

### How LangChain is Used

**LangChain** provides the foundation for our AI agent system:

1. **LLM Integration**: Each agent uses `ChatOpenAI` from LangChain to interact with GPT-4
2. **Agent Framework**: Agents use LangChain's agent framework with tools and memory
3. **Prompt Templates**: Structured prompts using `ChatPromptTemplate` and `MessagesPlaceholder`
4. **Memory Management**: Each agent has `ConversationBufferMemory` for context retention
5. **Tool Integration**: Custom tools for stock data, news fetching, and ticker lookup

### How LangGraph is Used

**LangGraph** orchestrates the entire workflow as a state machine:

1. **State Management**: `GraphState` tracks the complete workflow state
2. **Workflow Orchestration**: Sequential execution of agents through defined nodes
3. **Error Handling**: Robust error propagation and state recovery
4. **Parallel Processing**: Efficient execution of independent tasks

### Agent Workflow with State Management

```mermaid
graph TB
    A[User Query] --> B[Initialize GraphState]
    B --> C[Ticker Lookup Step]
    C --> D[Update WorkflowState]
    D --> E[Research Step]
    E --> F[Update StockData & NewsData]
    F --> G[Analysis Step]
    G --> H[Update TechnicalAnalysis]
    H --> I[Sentiment Step]
    I --> J[Update SentimentAnalysis]
    J --> K[Summarization Step]
    K --> L[Create Final Summary]
    L --> M[Return Response]
    
    subgraph "State Flow"
        N[GraphState] --> O[WorkflowState]
        O --> P[AgentStates]
        P --> Q[Task Tracking]
        Q --> R[Data Models]
        R --> S[Final Output]
    end
    
    subgraph "Error Handling"
        T[Retry Logic] --> U[State Recovery]
        U --> V[Graceful Degradation]
    end
    
    C -.-> N
    E -.-> O
    G -.-> P
    I -.-> Q
    K -.-> R
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (required)
- NewsAPI key (optional, for enhanced news coverage)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/ai-stock-agent.git
cd ai-stock-agent
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp env.example .env
# Edit .env with your API keys
```

5. **Required environment variables**:
```env
OPENAI_API_KEY=your_openai_api_key_here
NEWS_API_KEY=your_news_api_key_here  # Optional
```

### Usage

#### Option 1: FastAPI Server (Recommended)

Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API documentation at: `http://localhost:8000/docs`

#### Option 2: Command Line Interface

```bash
# Interactive mode
python app.py

# Direct query
python app.py --query "Tell me about Apple stock"

# Batch analysis
python app.py --batch "Apple,Microsoft,Google"

# Custom output format
python app.py --query "TSLA analysis" --format json

# Verbose output
python app.py --query "Amazon stock" --verbose
```

## 🔌 API Endpoints

### 1. Stock Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
-H "Content-Type: application/json" \
-d '{"query": "Tell me about Apple stock"}'
```

### 2. Batch Analysis
```bash
curl -X POST "http://localhost:8000/batch-analyze" \
-H "Content-Type: application/json" \
-d '{"queries": ["Apple stock", "Microsoft analysis", "Google trends"]}'
```

### 3. Health Check
```bash
curl "http://localhost:8000/health"
```

## 🤖 Agent Architecture

### 🎯 Coordinator Agent (LangGraph)
**Role**: Orchestrates the entire analysis workflow using LangGraph
**Responsibilities**:
- Manages agent communication and state through LangGraph state machine
- Coordinates sequential execution of specialized agents
- Handles error recovery and retry logic
- Optimizes workflow execution and resource management

```python
class CoordinatorAgent:
    def __init__(self):
        self.workflow = self._build_workflow()  # LangGraph StateGraph
        self.ticker_lookup_agent = TickerLookupAgent()
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.sentiment_agent = SentimentAgent()
        self.summarization_agent = SummarizationAgent()
        self.app = self.workflow.compile()  # Compile LangGraph workflow
```

### 🔍 Ticker Lookup Agent
**Role**: Resolves company names to stock ticker symbols
**Capabilities**:
- Company name parsing and normalization
- Ticker symbol resolution with high accuracy
- Fuzzy matching for ambiguous company names
- Suggestion generation for failed lookups
- AI-assisted company name extraction from natural language

```python
class TickerLookupAgent:
    def __init__(self):
        # No configuration required - simplified initialization
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    async def resolve_company_ticker(self, query: str) -> Dict[str, Any]:
        # AI-powered ticker resolution
        return await self._ai_assisted_lookup(query)
```

### 📊 Research Agent (Refactored Architecture)
**Role**: Comprehensive data collection using specialized tools
**Key Features**:
- **Tool-Based Architecture**: Uses specialized tools that directly access services
- **No Data Duplication**: Single source of truth for data fetching
- **Clean Separation**: Tools handle data fetching, agent handles orchestration

**Tool Architecture**:
```python
class StockDataTool(BaseTool):
    """Directly accesses StockDataFetcher service"""
    def __init__(self):
        self.fetcher = StockDataFetcher()
    
    def _run(self, ticker: str, company_name: str, days: int = 7):
        # Direct service access with model conversion
        data = self.fetcher.get_comprehensive_data(ticker, company_name, days)
        return self._convert_to_stock_data_model(data)

class NewsDataTool(BaseTool):
    """Directly accesses NewsFetcher service"""
    def __init__(self):
        self.fetcher = NewsFetcher()
    
    def _run(self, company_name: str, ticker: str, limit: int = 5):
        # Direct service access with model conversion
        articles = self.fetcher.get_company_news(company_name, ticker, limit)
        return self._convert_to_news_data_model(articles)
```

**Research Agent**:
```python
class ResearchAgent:
    def __init__(self):
        # Simplified - no configuration required
        self.tools = [StockDataTool(), NewsDataTool()]
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    async def research_company(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Use tools to fetch data (no duplicate methods)
        stock_tool = self._get_tool_by_name("stock_data_fetcher")
        news_tool = self._get_tool_by_name("news_data_fetcher")
        
        # Execute tools for data collection
        stock_result = await stock_tool._arun(ticker, company_name)
        news_result = await news_tool._arun(company_name, ticker)
        
        return self._compile_research_results(stock_result, news_result)
```

### 📈 Analysis Agent
**Role**: Performs comprehensive technical analysis
**Capabilities**:
- Price trend analysis and pattern recognition
- Technical indicators calculation
- Support/resistance level identification
- Volume analysis and momentum assessment
- Volatility and risk metric calculation

```python
class AnalysisAgent:
    def __init__(self):
        # Simplified initialization - no config required
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    async def analyze_stock(self, stock_data: StockData) -> Dict[str, Any]:
        # Comprehensive technical analysis using GPT-4
        analysis_input = self._prepare_analysis_input(stock_data)
        result = await self.llm.ainvoke([
            SystemMessage(content=self.prompt.messages[0].content),
            HumanMessage(content=f"Analyze: {analysis_input}")
        ])
        return await self._process_analysis_result(result, stock_data)
```

### 📰 Sentiment Agent
**Role**: News sentiment analysis and theme extraction
**Capabilities**:
- Multi-article sentiment analysis
- Financial news interpretation
- Key theme and topic extraction
- Market impact assessment
- Sentiment scoring and classification

```python
class SentimentAgent:
    def __init__(self):
        # Simplified initialization
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    async def analyze_sentiment(self, news_data: NewsData) -> Dict[str, Any]:
        # Analyze sentiment of multiple news articles
        sentiment_input = self._prepare_sentiment_input(news_data)
        result = await self.llm.ainvoke([
            SystemMessage(content=self.prompt.messages[0].content),
            HumanMessage(content=f"Analyze sentiment: {sentiment_input}")
        ])
        return await self._process_sentiment_result(result, news_data)
```

### 📝 Summarization Agent
**Role**: Creates comprehensive natural language summaries
**Capabilities**:
- Executive summary generation
- Multi-source data synthesis
- Risk assessment compilation
- Actionable insights extraction
- Professional report formatting

```python
class SummarizationAgent:
    def __init__(self):
        # Simplified initialization
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    
    async def create_summary(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        # Create comprehensive summary from all analysis data
        summary_input = self._prepare_summary_input(workflow_state)
        result = await self.llm.ainvoke([
            SystemMessage(content=self.prompt.messages[0].content),
            HumanMessage(content=f"Create summary: {summary_input}")
        ])
        return await self._process_summary_result(result, workflow_state)
```
## 🔧 Troubleshooting

### Common Issues

#### 1. **Agent Initialization Errors**
```bash
Error: Failed to initialize agents
Solution: Check OpenAI API key in .env file
```

#### 2. **FastAPI Server Issues**
```bash
Error: uvicorn: command not found
Solution: pip install uvicorn[standard]
```

#### 3. **Missing API Keys**
```bash
Error: OpenAI API key not found
Solution: Set OPENAI_API_KEY in .env file
```

#### 4. **News API Issues**
```bash
Error: News data unavailable
Solution: News API key is optional - system works without it
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn app:app --reload --log-level debug

# Monitor agent performance
curl http://localhost:8000/health
```
## 🎯 Roadmap

### Phase 3: Advanced Features
- [ ] **Real-time WebSocket streaming**
- [ ] **Multi-model agent support** (GPT-4, Claude, Gemini)
- [ ] **Advanced caching strategies**
- [ ] **Distributed agent processing**
- [ ] **Custom agent marketplace**

### Phase 4: Enterprise Features
- [ ] **Multi-tenant architecture**
- [ ] **Advanced analytics dashboard**
- [ ] **Custom model fine-tuning**
- [ ] **Enterprise SSO integration**
- [ ] **Audit logging and compliance**


**🎉 Ready to analyze stocks with AI agents? Start the FastAPI server and explore the interactive docs at http://localhost:8000/docs!**
