"""
Stock data fetching module using yfinance.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from utils import log_info, log_error, format_percentage, format_price, get_date_range

class StockDataFetcher:
    """Handles fetching and processing stock data from Yahoo Finance."""
    
    def __init__(self):
        self.cache = {}
        
    def get_stock_info(self, ticker: str, company_name: str) -> Optional[Dict[str, Any]]:
        """Get basic stock information."""
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            # Extract key information
            stock_info = {
                'symbol': info.get('symbol', ticker.upper()),
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'website': info.get('website', 'N/A'),
                'business_summary': info.get('longBusinessSummary', 'N/A')
            }
            
            log_info(f"Successfully fetched stock info for {company_name}")
            return stock_info
            
        except Exception as e:
            log_error(f"Error fetching stock info for {company_name}: {str(e)}")
            return None
    
    def get_price_data(self, ticker: str, company_name: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Get historical price data for the specified number of days."""
        try:
            start_date, end_date = get_date_range(days)
            
            stock = yf.Ticker(ticker.upper())
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                log_error(f"No price data found for {company_name}")
                return None
                
            log_info(f"Successfully fetched {len(hist)} days of price data for {company_name}")
            return hist
            
        except Exception as e:
            log_error(f"Error fetching price data for {company_name}: {str(e)}")
            return None
    
    def calculate_price_movements(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price movements and trends from historical data."""
        try:
            if price_data.empty:
                return {}
            
            # Get first and last prices
            first_price = price_data['Close'].iloc[0]
            last_price = price_data['Close'].iloc[-1]
            
            # Calculate percentage change
            price_change = last_price - first_price
            percentage_change = (price_change / first_price) * 100
            
            # Calculate daily changes
            daily_changes = price_data['Close'].pct_change().dropna() * 100
            
            # Calculate volatility (standard deviation of daily returns)
            volatility = daily_changes.std()
            
            # Determine trend
            if percentage_change > 2:
                trend = "Strong Upward"
            elif percentage_change > 0:
                trend = "Upward"
            elif percentage_change < -2:
                trend = "Strong Downward"
            elif percentage_change < 0:
                trend = "Downward"
            else:
                trend = "Sideways"
            
            # Calculate highs and lows
            period_high = price_data['High'].max()
            period_low = price_data['Low'].min()
            
            # Calculate average volume
            avg_volume = price_data['Volume'].mean()
            
            movements = {
                'first_price': first_price,
                'last_price': last_price,
                'price_change': price_change,
                'percentage_change': percentage_change,
                'trend': trend,
                'volatility': volatility,
                'period_high': period_high,
                'period_low': period_low,
                'avg_volume': avg_volume,
                'daily_changes': daily_changes.tolist(),
                'formatted_change': format_percentage(percentage_change),
                'formatted_price': format_price(last_price)
            }
            
            log_info(f"Successfully calculated price movements: {trend} trend, {percentage_change:.2f}% change")
            return movements
            
        except Exception as e:
            log_error(f"Error calculating price movements: {str(e)}")
            return {}
    
    def get_comprehensive_data(self, ticker: str, company_name: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive stock data including info, prices, and movements."""
        try:
            # Get stock info
            stock_info = self.get_stock_info(ticker, company_name)
            if not stock_info:
                return {}
            
            # Get price data
            price_data = self.get_price_data(ticker, company_name, days)
            if price_data is None:
                return stock_info
            
            # Calculate movements
            movements = self.calculate_price_movements(price_data)
            
            # Combine all data
            comprehensive_data = {
                **stock_info,
                'price_data': price_data,
                'movements': movements,
                'data_period_days': days,
                'last_updated': datetime.now().isoformat()
            }
            
            log_info(f"Successfully compiled comprehensive data for {company_name}")
            return comprehensive_data
            
        except Exception as e:
            log_error(f"Error getting comprehensive data for {company_name}: {str(e)}")
            return {}
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker exists and has data."""
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            # Check if we got valid info
            if not info or 'symbol' not in info:
                return False
                
            # Try to get recent data
            hist = stock.history(period="1d")
            return not hist.empty
            
        except Exception as e:
            log_error(f"Error validating ticker {ticker}: {str(e)}")
            return False

 