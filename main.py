#!/usr/bin/env python3
"""
Stock Summary Agent - CLI Interface & Core Application Logic
Command-line interface and core StockSummaryAgent implementation
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import application components
from agents.coordinator import CoordinatorAgent
from models import SystemConfig, AgentType, AgentConfig, StockSummary
from utils import setup_environment, log_info, log_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# MAIN APPLICATION CLASS
# ===============================================================================

class StockSummaryAgent:
    """Main application class for the Stock Summary Agent."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Stock Summary Agent."""
        self.config = config or {}
        self.coordinator = None
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system components."""
        try:
            # Setup environment
            setup_environment()
            
            # Initialize coordinator
            self.coordinator = CoordinatorAgent(self.config)
            
            log_info("Stock Summary Agent initialized successfully")
            
        except Exception as e:
            log_error(f"Failed to initialize Stock Summary Agent: {str(e)}")
            raise
    
    async def analyze_stock(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze a stock based on user query.
        
        Args:
            query: User query about a stock (e.g., "Tell me about Apple stock")
            session_id: Optional session ID for tracking
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            log_info(f"Analyzing stock query: {query}")
            
            # Process the query through the coordinator
            result = await self.coordinator.process_query(query, session_id)
            
            if result["success"]:
                log_info(f"Analysis completed successfully for session {result['session_id']}")
                return self._format_success_response(result)
            else:
                log_error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                return self._format_error_response(result)
                
        except Exception as e:
            error_msg = f"Analysis failed with exception: {str(e)}"
            log_error(error_msg)
            return self._format_error_response({"error": error_msg})
    
    def _format_success_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful analysis response."""
        final_summary = result.get("final_summary", {})
        workflow_state = result.get("workflow_state")
        
        # Extract key information
        response = {
            "success": True,
            "session_id": result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "company_name": final_summary.get("company_name", "Unknown"),
                "ticker": final_summary.get("ticker", "Unknown"),
                "executive_summary": final_summary.get("executive_summary", "No summary available"),
                "price_analysis": final_summary.get("price_analysis", "No price analysis available"),
                "news_sentiment": final_summary.get("news_sentiment", "No sentiment analysis available"),
                "technical_outlook": final_summary.get("technical_outlook", "No technical analysis available"),
                "risk_assessment": final_summary.get("risk_assessment", "No risk assessment available")
            },
            "metadata": {
                "step_count": result.get("step_count", 0),
                "errors": result.get("errors", []),
                "data_sources": []
            }
        }
        
        # Add data sources if available
        if workflow_state:
            if workflow_state.stock_data:
                response["metadata"]["data_sources"].append("Stock Data (yfinance)")
            if workflow_state.news_data:
                response["metadata"]["data_sources"].append("News Data")
        
        return response
    
    def _format_error_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response."""
        return {
            "success": False,
            "session_id": result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
            "error": result.get("error", "Unknown error"),
            "errors": result.get("errors", [])
        }
    
    async def get_supported_companies(self) -> List[str]:
        """Get list of supported companies."""
        try:
            from services.ticker_lookup import TickerLookup
            lookup = TickerLookup()
            
            # Get list of supported companies
            companies = list(lookup.common_tickers.keys())
            return sorted([company.title() for company in companies])
            
        except Exception as e:
            log_error(f"Error getting supported companies: {str(e)}")
            return []
    
    async def validate_company(self, company_name: str) -> Dict[str, Any]:
        """Validate if a company is supported."""
        try:
            from services.ticker_lookup import TickerLookup
            lookup = TickerLookup()
            
            ticker = lookup.lookup_ticker(company_name)
            if ticker:
                company_info = lookup.get_company_name(ticker)
                return {
                    "valid": True,
                    "ticker": ticker,
                    "company_name": company_info
                }
            else:
                suggestions = lookup.suggest_tickers(company_name, limit=3)
                return {
                    "valid": False,
                    "suggestions": suggestions
                }
                
        except Exception as e:
            log_error(f"Error validating company: {str(e)}")
            return {"valid": False, "error": str(e)}

# ===============================================================================
# CONVENIENCE FUNCTIONS
# ===============================================================================

async def analyze_stock_simple(query: str) -> Dict[str, Any]:
    """Simple function to analyze a stock with default configuration."""
    agent = StockSummaryAgent()
    return await agent.analyze_stock(query)

async def analyze_multiple_stocks(queries: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple stocks concurrently."""
    agent = StockSummaryAgent()
    
    # Create tasks for concurrent execution
    tasks = [agent.analyze_stock(query) for query in queries]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "query": queries[i],
                "error": str(result)
            })
        else:
            result["query"] = queries[i]
            processed_results.append(result)
    
    return processed_results

# ===============================================================================
# CLI APPLICATION CLASS
# ===============================================================================

class StockAgentApp:
    """CLI application class with clean entry points."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the application."""
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        self.agent = None
        self._setup_complete = False
        
    def setup(self) -> bool:
        """Setup the application environment and components."""
        try:
            log_info("🚀 Starting Stock Summary Agent")
            log_info("=" * 50)
            
            # Step 1: Setup environment
            log_info("📋 Step 1: Loading environment configuration...")
            setup_environment()
            
            # Step 2: Validate configuration
            log_info("🔍 Step 2: Validating configuration...")
            if not self._validate_config():
                return False
            
            # Step 3: Initialize agent
            log_info("🤖 Step 3: Initializing AI agents...")
            self.agent = StockSummaryAgent()
            
            log_info("✅ Setup completed successfully!")
            self._setup_complete = True
            return True
            
        except Exception as e:
            log_error(f"❌ Setup failed: {str(e)}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate configuration and API keys."""
        import os
        
        # Check for OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key or openai_key == 'your_openai_api_key_here':
            log_error("❌ OpenAI API key is required!")
            log_error("   Please set OPENAI_API_KEY in your .env file")
            log_error("   Get your API key from: https://platform.openai.com/api-keys")
            return False
        
        log_info("✅ OpenAI API key configured")
        
        # Check for optional News API key
        news_key = os.getenv('NEWS_API_KEY')
        if news_key and news_key != 'your_news_api_key_here':
            log_info("✅ News API key configured (enhanced news coverage)")
        else:
            log_info("ℹ️  News API key not configured (using free sources)")
        
        return True
    
    async def analyze_single_stock(self, query: str, output_format: str = 'text') -> Dict[str, Any]:
        """Analyze a single stock query."""
        if not self._setup_complete:
            raise RuntimeError("Application not setup. Call setup() first.")
        
        log_info(f"🔍 Analyzing: {query}")
        log_info("-" * 40)
        
        try:
            # Perform analysis
            result = await self.agent.analyze_stock(query)
            
            # Log summary
            if result['success']:
                analysis = result['analysis']
                log_info(f"✅ Analysis completed for {analysis['company_name']} ({analysis['ticker']})")
                log_info(f"📊 Processing steps: {result['metadata']['step_count']}")
                log_info(f"📈 Data sources: {', '.join(result['metadata']['data_sources'])}")
                
                if result['metadata']['errors']:
                    log_info(f"⚠️  Warnings: {len(result['metadata']['errors'])}")
            else:
                log_error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            log_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    async def analyze_multiple_stocks(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple stocks concurrently."""
        if not self._setup_complete:
            raise RuntimeError("Application not setup. Call setup() first.")
        
        log_info(f"🔍 Analyzing {len(queries)} stocks concurrently...")
        log_info("-" * 40)
        
        try:
            results = await analyze_multiple_stocks(queries)
            
            # Log summary
            successful = sum(1 for r in results if r.get('success', False))
            failed = len(results) - successful
            
            log_info(f"✅ Batch analysis completed: {successful} successful, {failed} failed")
            
            return results
            
        except Exception as e:
            error_msg = f"Batch analysis failed: {str(e)}"
            log_error(f"❌ {error_msg}")
            return [{'success': False, 'error': error_msg, 'query': q} for q in queries]
    
    def print_analysis_result(self, result: Dict[str, Any]):
        """Print analysis results in a formatted way."""
        if not result['success']:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return

        analysis = result['analysis']
        
        print("\n" + "=" * 60)
        print("📊 STOCK ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Company: {analysis['company_name']} ({analysis['ticker']})")
        print(f"Analysis Date: {result['timestamp']}")
        print()
        
        print("📋 EXECUTIVE SUMMARY")
        print("-" * 40)
        print(analysis["executive_summary"])
        print()
        
        print("📈 PRICE ANALYSIS")
        print("-" * 40)
        print(analysis["price_analysis"])
        print()
        
        print("📰 NEWS SENTIMENT")
        print("-" * 40)
        print(analysis["news_sentiment"])
        print()
        
        print("🔍 TECHNICAL OUTLOOK")
        print("-" * 40)
        print(analysis["technical_outlook"])
        print()
        
        print("⚠️  RISK ASSESSMENT")
        print("-" * 40)
        print(analysis["risk_assessment"])
        print()
        
        metadata = result["metadata"]
        print("ℹ️  METADATA")
        print("-" * 40)
        print(f"Processing Steps: {metadata['step_count']}")
        print(f"Data Sources: {', '.join(metadata['data_sources'])}")
        if metadata["errors"]:
            print(f"Warnings: {len(metadata['errors'])}")
        print()

# ===============================================================================
# COMMAND LINE INTERFACE
# ===============================================================================

async def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="Stock Summary Agent - AI-powered stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Tell me about Apple stock"
  %(prog)s "What's the outlook for Tesla?" --output json
  %(prog)s "Analyze Microsoft" --verbose
  %(prog)s --batch queries.txt
        """
    )
    
    parser.add_argument(
        "query", 
        nargs="?", 
        help="Stock query to analyze (e.g., 'Tell me about Apple stock')"
    )
    parser.add_argument(
        "--output", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--batch", 
        metavar="FILE",
        help="Batch analyze stocks from a file (one query per line)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.query and not args.batch:
        parser.error("Either provide a query or use --batch with a file")
    
    # Initialize application
    app = StockAgentApp(verbose=args.verbose)
    
    # Setup application
    if not app.setup():
        log_error("❌ Application setup failed. Exiting.")
        return 1
    
    try:
        if args.batch:
            # Batch processing
            log_info(f"📁 Reading queries from: {args.batch}")
            
            try:
                with open(args.batch, 'r') as f:
                    queries = [line.strip() for line in f if line.strip()]
                
                if not queries:
                    log_error("❌ No queries found in batch file")
                    return 1
                
                results = await app.analyze_multiple_stocks(queries)
                
                # Output results
                if args.output == "json":
                    print(json.dumps(results, indent=2, default=str))
                else:
                    for i, result in enumerate(results, 1):
                        print(f"\n{'='*20} ANALYSIS {i}/{len(results)} {'='*20}")
                        print(f"Query: {result.get('query', 'Unknown')}")
                        app.print_analysis_result(result)
                
            except FileNotFoundError:
                log_error(f"❌ Batch file not found: {args.batch}")
                return 1
                
        else:
            # Single query processing
            result = await app.analyze_single_stock(args.query, args.output)
            
            # Output result
            if args.output == "json":
                print(json.dumps(result, indent=2, default=str))
            else:
                app.print_analysis_result(result)
        
        log_info("🎉 Analysis completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        log_info("🛑 Analysis interrupted by user")
        return 1
    except Exception as e:
        log_error(f"❌ Application error: {str(e)}")
        return 1

# ===============================================================================
# ASYNC CONTEXT MANAGER
# ===============================================================================

class AsyncStockAgent:
    """Async context manager for the Stock Summary Agent."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config
        self.agent = None
    
    async def __aenter__(self):
        self.agent = StockSummaryAgent(self.config)
        return self.agent
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

# Example usage with async context manager
async def example_with_context_manager():
    """Example of using the async context manager."""
    async with AsyncStockAgent() as agent:
        result = await agent.analyze_stock("Tell me about Tesla")
        print(f"Analysis completed: {result['success']}")

# ===============================================================================
# BATCH PROCESSING UTILITIES
# ===============================================================================

async def batch_analyze_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Batch analyze stocks from a file."""
    try:
        with open(file_path, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = await analyze_multiple_stocks(queries)
        return results
        
    except Exception as e:
        log_error(f"Error in batch processing: {str(e)}")
        return []

async def save_results_to_file(results: List[Dict[str, Any]], output_path: str):
    """Save analysis results to a file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log_info(f"Results saved to {output_path}")
        
    except Exception as e:
        log_error(f"Error saving results: {str(e)}")

# ===============================================================================
# MAIN ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    # If called directly, run the command line interface
    sys.exit(asyncio.run(main())) 