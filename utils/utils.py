"""
Utility functions for the stock summary agent.
"""
import os
import re
import inspect
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
    
def get_env_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.getenv(var_name, default)

def validate_ticker(ticker: str) -> bool:
    """Validate if ticker format is correct."""
    # Basic validation: 1-5 uppercase letters
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, ticker.upper()))

def clean_company_name(company_name: str) -> str:
    """Clean and normalize company name for lookup."""
    # Remove common suffixes and normalize
    cleaned = company_name.strip()
    # Remove Inc., Corp., Ltd., etc.
    suffixes = ['Inc.', 'Corp.', 'Ltd.', 'LLC', 'Co.', 'Company']
    for suffix in suffixes:
        cleaned = re.sub(rf'\b{suffix}\b', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

def format_percentage(value: float) -> str:
    """Format percentage with appropriate color coding info."""
    if value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"

def format_price(price: float) -> str:
    """Format price with currency symbol."""
    return f"${price:.2f}"

def get_date_range(days: int = 7) -> tuple[datetime, datetime]:
    """Get date range for the last N days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

def safe_get(data: Dict[Any, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default."""
    try:
        return data.get(key, default)
    except (AttributeError, KeyError):
        return default

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def log_info(message: str):
    """Log info message with caller's module name."""
    # Get the caller's frame to determine the calling module
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'unknown')
    logger = logging.getLogger(module_name)
    logger.info(message)

def log_error(message: str):
    """Log error message with caller's module name."""
    # Get the caller's frame to determine the calling module
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'unknown')
    logger = logging.getLogger(module_name)
    logger.error(message) 