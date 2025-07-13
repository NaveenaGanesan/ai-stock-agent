# AI Stock Agent

A sophisticated AI-powered stock analysis platform featuring a complete agent-based architecture with LangChain/LangGraph orchestration. The system provides both FastAPI REST endpoints and a command-line interface for comprehensive stock market analysis.

## 🏗️ Architecture Overview

**Complete Agent-Based System**: Every component is an intelligent agent that collaborates through LangGraph workflows:

```mermaid
graph TB
    subgraph "Client Layer"
        A[REST API Client] --> B[FastAPI Backend]
        C[CLI Client] --> D[CLI Interface]
    end
    
    subgraph "Agent Orchestration Layer"
        B --> E[Coordinator Agent]
        D --> E
        E --> F[LangGraph Workflow]
    end
    
    subgraph "Specialized Agents"
        F --> G[Research Agent]
        F --> H[Analysis Agent]
        F --> I[Sentiment Agent]
        F --> J[Summary Agent]
    end
    
    subgraph "Data Sources"
        G --> K[Stock Data API]
        G --> L[News API]
        G --> M[Ticker Lookup]
    end
```

## 📁 Project Structure

```
ai-stock-agent/
├── app.py                           # Unified FastAPI Backend + CLI Interface
├── models/                          # Data Models & Schemas
│   ├── __init__.py
│   └── models.py                   # Unified StockAgentModels class
├── agents/                          # AI Agents & Orchestration
│   ├── __init__.py
│   ├── coordinator.py              # LangGraph workflow coordinator
│   ├── ticker_lookup_agent.py      # Company/ticker resolution agent
│   ├── research_agent.py           # Data collection & research agent
│   ├── analysis_agent.py           # Technical analysis agent
│   ├── sentiment_agent.py          # News sentiment analysis agent
│   └── summarization_agent.py      # Final summary generation agent
├── services/                        # External Services Integration
│   ├── __init__.py
│   ├── ticker_lookup.py            # Company/ticker resolution service
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

### Agent Workflow

```mermaid
graph TB
    A[Initialize] --> B[Ticker Lookup Agent]
    B --> C[Research Agent]
    C --> D[Analysis Agent]
    D --> E[Sentiment Agent]
    E --> F[Summarization Agent]
    F --> G[Finalize]
    
    subgraph "LangGraph State Machine"
        H[GraphState] --> I[WorkflowState]
        I --> J[Agent Responses]
        J --> K[Final Summary]
    end
```

### Individual Agent Architecture

Each agent is a self-contained unit with:
- **LangChain LLM**: GPT-4 integration with specific temperature settings
- **Custom Tools**: Agent-specific tools for data fetching and processing
- **Prompt Engineering**: Specialized prompts for optimal performance
- **Error Handling**: Retry logic and graceful failure handling
- **State Management**: Local state tracking and workflow integration

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required API keys (see Configuration section)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-stock-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp config/env.example .env
   # Edit .env with your API keys
   ```

## 🔧 Configuration

Copy `config/env.example` to `.env` and configure:

```env
# Core Configuration
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-stock-agent

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_RELOAD=true

# Stock Data Configuration
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here
POLYGON_API_KEY=your_polygon_key_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here
GNEWS_API_KEY=your_gnews_key_here
```

## 🎯 How to Start the Application

### Option 1: FastAPI Backend (REST API)

Start the FastAPI server:

```bash
# Start the FastAPI server
python app.py --server

# Or specify custom host/port
python app.py --server --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health
- **Analysis Endpoint**: http://localhost:8000/analyze

### Option 2: CLI Interface

Use the command-line interface:

```bash
# Analyze a single stock
python app.py --query "Tell me about Apple stock"

# Analyze multiple stocks
python app.py --batch "Apple,Microsoft,Google"

# Custom output format
python app.py --query "TSLA analysis" --format json

# Verbose output
python app.py --query "Amazon stock" --verbose

# Interactive mode (default)
python app.py
```

### Option 3: Python Library

Use as a Python library:

```python
import asyncio
from app import analyze_stock_simple

async def main():
    result = await analyze_stock_simple("Tell me about Apple stock")
    print(result)

asyncio.run(main())
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

### 3. Ticker Validation
```bash
curl -X POST "http://localhost:8000/validate-ticker" \
-H "Content-Type: application/json" \
-d '{"company_name": "Apple"}'
```

### 4. Supported Companies
```bash
curl "http://localhost:8000/supported-companies"
```

### 5. Health Check
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
    async def resolve_company_ticker(self, query: str) -> Dict[str, Any]:
        # Direct lookup first, then AI-assisted extraction
        result = await self._try_direct_lookup(query)
        if not result["success"]:
            result = await self._ai_assisted_lookup(query)
        return result
```

### 📊 Research Agent
**Role**: Comprehensive data collection and validation
**Capabilities**:
- Stock price data fetching (yfinance integration)
- News article collection from multiple sources
- Data quality validation and cleaning
- Historical data analysis
- Real-time market data integration

```python
class ResearchAgent:
    def __init__(self):
        self.tools = [StockDataTool(), NewsDataTool(), TickerLookupTool()]
        self.stock_fetcher = StockDataFetcher()
        self.news_fetcher = NewsFetcher()
        
    async def research_company(self, query: str) -> Dict[str, Any]:
        # Fetch comprehensive data for analysis
        return await self.execute_task(task, workflow_state)
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
    async def create_summary(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        # Create comprehensive summary from all analysis data
        summary_input = self._prepare_summary_input(workflow_state)
        result = await self.llm.ainvoke([
            SystemMessage(content=self.prompt.messages[0].content),
            HumanMessage(content=f"Create summary: {summary_input}")
        ])
        return await self._process_summary_result(result, workflow_state)
```

### 🔧 LangGraph Workflow Integration

Each agent integrates seamlessly with the LangGraph workflow:

```python
# LangGraph workflow definition
workflow = StateGraph(GraphState)
workflow.add_node("initialize", self._initialize_workflow)
workflow.add_node("ticker_lookup", self._ticker_lookup_step)
workflow.add_node("research", self._research_step)
workflow.add_node("analysis", self._analysis_step)
workflow.add_node("sentiment", self._sentiment_step)
workflow.add_node("summarization", self._summarization_step)
workflow.add_node("finalize", self._finalize_workflow)

# Sequential execution flow
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "ticker_lookup")
workflow.add_edge("ticker_lookup", "research")
workflow.add_edge("research", "analysis")
workflow.add_edge("analysis", "sentiment")
workflow.add_edge("sentiment", "summarization")
workflow.add_edge("summarization", "finalize")
workflow.add_edge("finalize", END)

# Compile and execute
app = workflow.compile()
result = await app.ainvoke(initial_state)
```

### 🔄 Workflow Execution Flow

1. **Initialize**: Set up workflow state and prepare for execution
2. **Ticker Lookup**: Resolve company name to ticker symbol using AI
3. **Research**: Fetch stock data and news articles
4. **Analysis**: Perform technical analysis on stock data
5. **Sentiment**: Analyze news sentiment and extract themes
6. **Summarization**: Create comprehensive final summary
7. **Finalize**: Clean up and prepare final output

### 🧠 AI Integration Details

- **GPT-4 Model**: All agents use GPT-4 for intelligent analysis
- **Specialized Prompts**: Each agent has custom prompts for optimal performance
- **Temperature Control**: Different temperature settings for different tasks
- **Memory Management**: Conversation history for context retention
- **Error Recovery**: Automatic retry logic with exponential backoff

### 📊 Data Flow

```
User Query → Ticker Lookup → Stock Data + News Data → Analysis + Sentiment → Final Summary
```

Each step enriches the workflow state with additional data:
- **Ticker & Company**: Resolved identifiers
- **Stock Data**: Price history, movements, technical indicators
- **News Data**: Recent articles, publication dates, sources
- **Analysis Results**: Technical insights, trends, support/resistance
- **Sentiment Results**: Overall sentiment, themes, market impact
- **Final Summary**: Comprehensive analysis report

## 🔧 Agent Data Models

### 📊 Core Data Models
```python
class AgentState(BaseModel):
    """Global state shared across all agents"""
    session_id: str
    query: str
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    collected_data: Optional[CollectedData] = None
    analyses: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[StockSummary] = None
    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

374|
class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    async def analyze_technical(self, stock_data: StockData) -> TechnicalAnalysis:
        # Calculate technical indicators
        indicators = self.indicators.calculate_all(stock_data)
        
        # AI-powered analysis using LLM
        analysis = await self.llm.ainvoke(self._create_analysis_prompt(indicators))
        
        return TechnicalAnalysis.from_llm_response(analysis)
```

### 📰 Sentiment Analysis Agent
**Role**: Analyzes market sentiment from news and social media
**Capabilities**:
- News sentiment analysis
- Social media sentiment tracking
- Market mood assessment
- Sentiment trend analysis

```python
class SentimentAnalysisAgent(BaseAgent):
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    async def analyze_sentiment(self, news_data: NewsData) -> SentimentAnalysis:
        # Analyze individual article sentiments
        article_sentiments = await self.sentiment_analyzer.analyze_batch(news_data.articles)
        
        # Aggregate and contextualize using LLM
        overall_sentiment = await self.llm.ainvoke(
            self._create_sentiment_prompt(article_sentiments)
        )
        
        return SentimentAnalysis.from_aggregated_data(overall_sentiment)
```

### 🎯 Risk Assessment Agent
**Role**: Evaluates investment risks and opportunities
**Capabilities**:
- Financial risk analysis
- Market risk assessment
- Regulatory risk evaluation
- Opportunity identification

```python
class RiskAssessmentAgent(BaseAgent):
    def __init__(self):
        self.risk_models = RiskModels()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    
    async def assess_risk(self, analysis_data: AnalysisData) -> RiskAssessment:
        # Calculate risk metrics
        risk_metrics = self.risk_models.calculate_risk_metrics(analysis_data)
        
        # AI-powered risk assessment
        risk_analysis = await self.llm.ainvoke(
            self._create_risk_prompt(risk_metrics)
        )
        
        return RiskAssessment.from_analysis(risk_analysis)
```

### 📝 Summarization Agent
**Role**: Creates comprehensive natural language summaries
**Capabilities**:
- Executive summary generation
- Key insights extraction
- Recommendation synthesis
- Multi-format output (text, structured data)

```python
class SummarizationAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    
    async def create_summary(self, all_analyses: CompleteAnalysis) -> StockSummary:
        # Generate comprehensive summary using all analysis data
        summary = await self.llm.ainvoke(
            self._create_summary_prompt(all_analyses)
        )
        
        return StockSummary.from_llm_summary(summary)
```

## 🔧 Agent Data Models

### 📊 Core Data Models
```python
class AgentState(BaseModel):
    """Global state shared across all agents"""
    session_id: str
    query: str
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    collected_data: Optional[CollectedData] = None
    analyses: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[StockSummary] = None
    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

class CollectedData(BaseModel):
    """Data collected by DataCollectionAgent"""
    ticker: str
    stock_data: StockData
    news_data: NewsData
    market_data: MarketData
    collection_timestamp: datetime = Field(default_factory=datetime.now)

class TechnicalAnalysis(BaseModel):
    """Technical analysis results"""
    trend_direction: TrendDirection
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    key_indicators: Dict[str, float]
    technical_summary: str

class SentimentAnalysis(BaseModel):
    """Sentiment analysis results"""
    overall_sentiment: SentimentType
    sentiment_score: float  # -1 to 1
    news_sentiment: NewsSentiment
    market_mood: str
    sentiment_trends: List[SentimentTrend]

class RiskAssessment(BaseModel):
    """Risk assessment results"""
    overall_risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    opportunities: List[Opportunity]
    risk_score: float  # 0 to 1
    recommendations: List[str]

class StockSummary(BaseModel):
    """Final comprehensive summary"""
    company_name: str
    ticker: str
    executive_summary: str
    price_analysis: str
    sentiment_analysis: str
    risk_assessment: str
    recommendations: List[str]
    confidence_score: float
```

## 🔐 Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_WORKERS=4
FASTAPI_RELOAD=false

# Agent Configuration
COORDINATOR_AGENT_TEMPERATURE=0.3
TECHNICAL_AGENT_TEMPERATURE=0.3
SENTIMENT_AGENT_TEMPERATURE=0.5
RISK_AGENT_TEMPERATURE=0.4
SUMMARY_AGENT_TEMPERATURE=0.6

# System Configuration
MAX_CONCURRENT_REQUESTS=10
AGENT_TIMEOUT_SECONDS=30
MAX_RETRY_ATTEMPTS=3
CACHE_TTL_MINUTES=15
```

## 🚀 Deployment

### 🐳 Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t stock-agent .
docker run -p 8000:8000 --env-file .env stock-agent
```

### ☁️ Production Deployment
```bash
# Using gunicorn for production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using systemd service
sudo systemctl enable stock-agent
sudo systemctl start stock-agent
```

## 🧪 Testing

### Unit Tests
```bash
# Test individual agents
python -m pytest tests/test_agents.py -v

# Test FastAPI endpoints
python -m pytest tests/test_api.py -v

# Test agent integration
python -m pytest tests/test_integration.py -v
```

### API Testing
```bash
# Test API endpoints
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test Apple stock"}'

# Load testing
ab -n 100 -c 10 http://localhost:8000/health
```

## 📊 Monitoring & Logging

### 📈 Metrics
- **Agent Performance**: Response times, success rates
- **API Metrics**: Request rates, error rates
- **Resource Usage**: Memory, CPU, API calls
- **Business Metrics**: Analysis accuracy, user satisfaction

### 📋 Logging
```python
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Stock Summary Agent", version="2.0.0")

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"Request: {request.method} {request.url} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.2f}s")
    
    return response
```

## 🔍 Usage Examples

### 🐍 Python Client
```python
import httpx
import asyncio

class StockAnalysisClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def analyze_stock(self, query: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/analyze",
                json={"query": query}
            )
            return response.json()
    
    async def batch_analyze(self, queries: list) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/batch-analyze",
                json={"queries": queries}
            )
            return response.json()

# Usage
client = StockAnalysisClient()
result = asyncio.run(client.analyze_stock("Tell me about Apple stock"))
print(result["analysis"]["executive_summary"])
```

### 🌐 JavaScript Client
```javascript
class StockAnalysisClient {
    constructor(baseUrl = "http://localhost:8000") {
        this.baseUrl = baseUrl;
    }
    
    async analyzeStock(query) {
        const response = await fetch(`${this.baseUrl}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        return await response.json();
    }
    
    async batchAnalyze(queries) {
        const response = await fetch(`${this.baseUrl}/batch-analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ queries })
        });
        return await response.json();
    }
}

// Usage
const client = new StockAnalysisClient();
const result = await client.analyzeStock("Tell me about Apple stock");
console.log(result.analysis.executive_summary);
```

### 📱 cURL Examples
```bash
# Single stock analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about Apple stock",
    "options": {
      "include_technical": true,
      "include_sentiment": true,
      "days_history": 30
    }
  }'

# Batch analysis
curl -X POST "http://localhost:8000/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "Apple stock analysis",
      "Tesla outlook",
      "Microsoft performance"
    ],
    "options": {
      "concurrent": true
    }
  }'

# Validate ticker
curl -X POST "http://localhost:8000/validate-ticker" \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Apple Inc."}'
```

## 🛡️ Security & Rate Limiting

### 🔐 Authentication
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials
```

### 📊 Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_stock(request: Request, query: AnalysisRequest):
    # Analysis logic
    pass
```

## 📈 Performance Optimization

### 🚀 Caching
```python
from functools import lru_cache
import redis

# Redis cache for API responses
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=100)
def get_stock_data_cached(ticker: str, days: int):
    cache_key = f"stock_data:{ticker}:{days}"
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # Fetch new data
    data = fetch_stock_data(ticker, days)
    redis_client.setex(cache_key, 300, json.dumps(data))  # 5 min TTL
    return data
```

### ⚡ Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncAgentManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_parallel(self, tasks: list):
        # Run CPU-intensive tasks in thread pool
        loop = asyncio.get_event_loop()
        results = await asyncio.gather(*[
            loop.run_in_executor(self.executor, task)
            for task in tasks
        ])
        return results
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **Agent Initialization Errors**
```bash
Error: Failed to initialize CoordinatorAgent
Solution: Check OpenAI API key and model permissions
```

#### 2. **FastAPI Server Issues**
```bash
Error: uvicorn: command not found
Solution: pip install uvicorn[standard]
```

#### 3. **Agent Timeout Errors**
```bash
Error: Agent execution timeout
Solution: Increase AGENT_TIMEOUT_SECONDS in .env
```

#### 4. **Rate Limiting**
```bash
Error: Rate limit exceeded
Solution: Implement proper rate limiting and retry logic
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn main:app --reload --log-level debug

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

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**🎉 Ready to analyze stocks with AI agents? Start the FastAPI server and explore the interactive docs at http://localhost:8000/docs!**
