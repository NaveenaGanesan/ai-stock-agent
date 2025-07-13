"""
Ticker lookup module to map company names to stock tickers.
"""
import yfinance as yf
import requests
from typing import Optional, Dict, List, Any
from utils import log_info, log_error, clean_company_name
import json

class TickerLookup:
    """Handles mapping company names to stock tickers."""
    
    def __init__(self):
        # Common company-to-ticker mappings for quick lookup
        self.common_tickers = {
            # Tech Giants
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'meta': 'META',
            'facebook': 'META',
            'tesla': 'TSLA',
            'netflix': 'NFLX',
            'nvidia': 'NVDA',
            'salesforce': 'CRM',
            'adobe': 'ADBE',
            'oracle': 'ORCL',
            'intel': 'INTC',
            'cisco': 'CSCO',
            'ibm': 'IBM',
            
            # Financial
            'jpmorgan': 'JPM',
            'jp morgan': 'JPM',
            'bank of america': 'BAC',
            'wells fargo': 'WFC',
            'goldman sachs': 'GS',
            'morgan stanley': 'MS',
            'american express': 'AXP',
            'visa': 'V',
            'mastercard': 'MA',
            'paypal': 'PYPL',
            'square': 'SQ',
            'berkshire hathaway': 'BRK.A',
            
            # Healthcare
            'johnson & johnson': 'JNJ',
            'pfizer': 'PFE',
            'merck': 'MRK',
            'abbvie': 'ABBV',
            'bristol myers squibb': 'BMY',
            'eli lilly': 'LLY',
            'moderna': 'MRNA',
            'regeneron': 'REGN',
            'gilead': 'GILD',
            'amgen': 'AMGN',
            
            # Consumer
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'procter & gamble': 'PG',
            'walmart': 'WMT',
            'target': 'TGT',
            'home depot': 'HD',
            'nike': 'NKE',
            'starbucks': 'SBUX',
            'mcdonalds': 'MCD',
            'disney': 'DIS',
            
            # Energy
            'exxon mobil': 'XOM',
            'chevron': 'CVX',
            'conocophillips': 'COP',
            'schlumberger': 'SLB',
            
            # Industrial
            'boeing': 'BA',
            'caterpillar': 'CAT',
            'general electric': 'GE',
            '3m': 'MMM',
            'honeywell': 'HON',
            'lockheed martin': 'LMT',
            
            # Automotive
            'ford': 'F',
            'general motors': 'GM',
            'ferrari': 'RACE',
            
            # Crypto-related
            'coinbase': 'COIN',
            'microstrategy': 'MSTR',
            
            # Communication
            'verizon': 'VZ',
            'at&t': 'T',
            'comcast': 'CMCSA',
            'sprint': 'S',
            
            # Real Estate
            'american tower': 'AMT',
            'prologis': 'PLD',
            'simon property group': 'SPG',
        }
        
        # Create reverse mapping for ticker to name
        self.ticker_to_name = {v: k for k, v in self.common_tickers.items()}
    
    def lookup_ticker(self, company_name: str) -> Optional[str]:
        """
        Look up ticker symbol for a given company name.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Ticker symbol if found, None otherwise
        """
        try:
            # Clean the company name
            cleaned_name = clean_company_name(company_name).lower()
            
            # First, check our common mappings
            if cleaned_name in self.common_tickers:
                ticker = self.common_tickers[cleaned_name]
                log_info(f"Found ticker {ticker} for {company_name} in common mappings")
                return ticker
            
            # Check for partial matches in common mappings
            for name, ticker in self.common_tickers.items():
                if cleaned_name in name or name in cleaned_name:
                    log_info(f"Found partial match: {ticker} for {company_name}")
                    return ticker
            
            # If not found in common mappings, try yfinance search
            ticker = self._search_yfinance(company_name)
            if ticker:
                return ticker
            
            # If still not found, try alternative search methods
            ticker = self._search_alternative(company_name)
            if ticker:
                return ticker
            
            log_error(f"Could not find ticker for company: {company_name}")
            return None
            
        except Exception as e:
            log_error(f"Error looking up ticker for {company_name}: {str(e)}")
            return None
    
    def _search_yfinance(self, company_name: str) -> Optional[str]:
        """Search for ticker using yfinance."""
        try:
            # Try to create ticker object and see if it's valid
            # This is a simple approach - in practice, yfinance doesn't have
            # a direct search API, so we'll try some common patterns
            
            # Try the company name as is (for cases like "AAPL")
            if self._validate_ticker_exists(company_name):
                return company_name.upper()
            
            # Try adding common stock exchange suffixes
            suffixes = ['', '.US', '.TO', '.L']
            for suffix in suffixes:
                test_ticker = company_name.upper() + suffix
                if self._validate_ticker_exists(test_ticker):
                    return test_ticker
            
            return None
            
        except Exception as e:
            log_error(f"Error in yfinance search for {company_name}: {str(e)}")
            return None
    
    def _search_alternative(self, company_name: str) -> Optional[str]:
        """Alternative search methods for ticker lookup."""
        try:
            # This is a placeholder for additional search methods
            # Could implement:
            # - Alpha Vantage search
            # - Financial Modeling Prep search
            # - SEC EDGAR search
            # - Web scraping financial sites
            
            log_info(f"Alternative search not implemented for {company_name}")
            return None
            
        except Exception as e:
            log_error(f"Error in alternative search for {company_name}: {str(e)}")
            return None
    
    def _validate_ticker_exists(self, ticker: str) -> bool:
        """Validate if a ticker exists using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid info
            if not info or len(info) <= 1:
                return False
            
            # Check if it has a symbol or name
            if 'symbol' not in info and 'longName' not in info:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_company_name(self, ticker: str) -> Optional[str]:
        """Get company name from ticker."""
        try:
            # Check reverse mapping first
            if ticker.upper() in self.ticker_to_name:
                return self.ticker_to_name[ticker.upper()].title()
            
            # Use yfinance to get company name
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            name = info.get('longName', info.get('shortName', None))
            if name:
                log_info(f"Found company name {name} for ticker {ticker}")
                return name
            
            return None
            
        except Exception as e:
            log_error(f"Error getting company name for {ticker}: {str(e)}")
            return None
    
    def suggest_tickers(self, partial_name: str, limit: int = 5) -> List[Dict[str, str]]:
        """Suggest possible tickers based on partial company name."""
        try:
            suggestions = []
            partial_lower = partial_name.lower()
            
            # Search through common mappings
            for name, ticker in self.common_tickers.items():
                if partial_lower in name:
                    suggestions.append({
                        'ticker': ticker,
                        'name': name.title(),
                        'match_type': 'partial'
                    })
            
            # Sort by relevance (exact matches first, then partial)
            suggestions.sort(key=lambda x: (
                x['match_type'] != 'exact',
                len(x['name']),
                x['name']
            ))
            
            return suggestions[:limit]
            
        except Exception as e:
            log_error(f"Error suggesting tickers for {partial_name}: {str(e)}")
            return []
    
    def add_mapping(self, company_name: str, ticker: str):
        """Add a new company-ticker mapping."""
        try:
            cleaned_name = clean_company_name(company_name).lower()
            self.common_tickers[cleaned_name] = ticker.upper()
            self.ticker_to_name[ticker.upper()] = cleaned_name
            log_info(f"Added mapping: {company_name} -> {ticker}")
            
        except Exception as e:
            log_error(f"Error adding mapping {company_name} -> {ticker}: {str(e)}")

 