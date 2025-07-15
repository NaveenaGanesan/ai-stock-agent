"""
News fetching module for retrieving recent financial news and headlines.
"""
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from utils import log_info, log_error, get_env_variable, truncate_text
import json
import time

class NewsFetcher:
    """Handles fetching financial news from various sources."""
    
    def __init__(self):
        self.news_api_key = get_env_variable("NEWS_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # RSS feeds for financial news
        self.rss_feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }
    
    def get_company_news(self, company_name: str, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent news for a specific company.
        
        Args:
            company_name: Name of the company
            ticker: Stock ticker symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        try:
            all_news = []
            
            # Try different sources
            sources = [
                self._get_yahoo_finance_news,
                self._get_newsapi_news,
                self._get_rss_news,
                self._get_google_news
            ]
            
            for source_func in sources:
                try:
                    news = source_func(company_name, ticker, limit)
                    if news:
                        all_news.extend(news)
                        log_info(f"Got {len(news)} articles from {source_func.__name__}")
                except Exception as e:
                    log_error(f"Error from {source_func.__name__}: {str(e)}")
                    continue
            
            # Remove duplicates and sort by date
            unique_news = self._deduplicate_news(all_news)
            sorted_news = sorted(unique_news, key=lambda x: x.get('published_date', datetime(1900, 1, 1)), reverse=True)
            
            # Return top articles
            result = sorted_news[:limit]
            log_info(f"Returning {len(result)} unique articles for {company_name}")
            
            return result
            
        except Exception as e:
            log_error(f"Error getting news for {company_name}: {str(e)}")
            return []
    
    def _get_yahoo_finance_news(self, company_name: str, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get news from Yahoo Finance."""
        try:
            # Yahoo Finance news URL
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            
            news_articles = []
            for entry in feed.entries[:limit]:
                try:
                    article = {
                        'title': entry.title,
                        'url': entry.link,
                        'summary': truncate_text(entry.get('summary', ''), 200),
                        'published_date': self._parse_date(entry.get('published', '')),
                        'source': 'Yahoo Finance',
                        'ticker': ticker
                    }
                    news_articles.append(article)
                except Exception as e:
                    log_error(f"Error parsing Yahoo Finance entry: {str(e)}")
                    continue
            
            return news_articles
            
        except Exception as e:
            log_error(f"Error fetching Yahoo Finance news: {str(e)}")
            return []
    
    def _get_newsapi_news(self, company_name: str, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get news from NewsAPI."""
        try:
            if not self.news_api_key:
                log_info("NewsAPI key not configured, skipping NewsAPI")
                return []
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Search query
            query = f'"{company_name}" OR "{ticker}" AND (stock OR shares OR market OR financial OR earnings)'
            
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'language': 'en',
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            # Handle specific HTTP error codes
            if response.status_code == 401:
                log_error("NewsAPI: Invalid API key, skipping NewsAPI for future requests")
                return []
            elif response.status_code == 429:
                log_error("NewsAPI: Rate limit exceeded, skipping this request")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            news_articles = []
            for article in data.get('articles', []):
                try:
                    news_article = {
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'summary': truncate_text(article.get('description', ''), 200),
                        'published_date': self._parse_date(article.get('publishedAt', '')),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'ticker': ticker
                    }
                    news_articles.append(news_article)
                except Exception as e:
                    log_error(f"Error parsing NewsAPI article: {str(e)}")
                    continue
            
            return news_articles
            
        except Exception as e:
            log_error(f"Error fetching NewsAPI news: {str(e)}")
            return []
    
    def _get_rss_news(self, company_name: str, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get news from RSS feeds and filter for company mentions."""
        try:
            all_articles = []
            
            for source_name, rss_url in self.rss_feeds.items():
                try:
                    response = self.session.get(rss_url, timeout=10)
                    response.raise_for_status()
                    
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries:
                        # Check if article mentions the company or ticker
                        title = entry.get('title', '').lower()
                        summary = entry.get('summary', '').lower()
                        
                        if (company_name.lower() in title or 
                            company_name.lower() in summary or
                            ticker.lower() in title or
                            ticker.lower() in summary):
                            
                            try:
                                article = {
                                    'title': entry.title,
                                    'url': entry.link,
                                    'summary': truncate_text(entry.get('summary', ''), 200),
                                    'published_date': self._parse_date(entry.get('published', '')),
                                    'source': source_name.replace('_', ' ').title(),
                                    'ticker': ticker
                                }
                                all_articles.append(article)
                            except Exception as e:
                                log_error(f"Error parsing RSS entry: {str(e)}")
                                continue
                    
                    # Add small delay between requests
                    time.sleep(0.5)
                    
                except Exception as e:
                    log_error(f"Error fetching RSS from {source_name}: {str(e)}")
                    continue
            
            # Return most recent articles
            sorted_articles = sorted(all_articles, key=lambda x: x.get('published_date', datetime(1900, 1, 1)), reverse=True)
            return sorted_articles[:limit]
            
        except Exception as e:
            log_error(f"Error fetching RSS news: {str(e)}")
            return []
    
    def _get_google_news(self, company_name: str, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get news from Google News RSS."""
        try:
            # Google News RSS URL
            query = f"{company_name} {ticker} stock"
            url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            news_articles = []
            for entry in feed.entries[:limit]:
                try:
                    article = {
                        'title': entry.title,
                        'url': entry.link,
                        'summary': truncate_text(entry.get('summary', ''), 200),
                        'published_date': self._parse_date(entry.get('published', '')),
                        'source': 'Google News',
                        'ticker': ticker
                    }
                    news_articles.append(article)
                except Exception as e:
                    log_error(f"Error parsing Google News entry: {str(e)}")
                    continue
            
            return news_articles
            
        except Exception as e:
            log_error(f"Error fetching Google News: {str(e)}")
            return []
    
    def _parse_date(self, date_string: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            if not date_string:
                return datetime.min
            
            # Common date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',  # RSS format with timezone
                '%a, %d %b %Y %H:%M:%S GMT', # RSS format with GMT
                '%Y-%m-%dT%H:%M:%SZ',        # ISO format
                '%Y-%m-%d %H:%M:%S',         # Simple format
                '%Y-%m-%d',                  # Date only
                '%a, %d %b %Y %H:%M:%S',     # RSS without timezone
            ]
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_string, fmt)
                    # Convert timezone-aware dates to naive (remove timezone info)
                    if parsed_date.tzinfo is not None:
                        parsed_date = parsed_date.replace(tzinfo=None)
                    return parsed_date
                except ValueError:
                    continue
            
            # If no format matches, return current time
            log_error(f"Could not parse date: {date_string}")
            return datetime.now()
            
        except Exception as e:
            log_error(f"Error parsing date {date_string}: {str(e)}")
            return datetime.min
    
    def _deduplicate_news(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate news articles based on title similarity."""
        try:
            if not news_list:
                return []
            
            unique_articles = []
            seen_titles = set()
            
            for article in news_list:
                title = article.get('title', '').lower().strip()
                
                # Simple deduplication based on title
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)
            
            return unique_articles
            
        except Exception as e:
            log_error(f"Error deduplicating news: {str(e)}")
            return news_list
    
    def get_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get general market news."""
        try:
            all_news = []
            
            # Get from multiple RSS feeds
            for source_name, rss_url in self.rss_feeds.items():
                try:
                    response = self.session.get(rss_url, timeout=10)
                    response.raise_for_status()
                    
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries[:5]:  # Limit per source
                        try:
                            article = {
                                'title': entry.title,
                                'url': entry.link,
                                'summary': truncate_text(entry.get('summary', ''), 200),
                                'published_date': self._parse_date(entry.get('published', '')),
                                'source': source_name.replace('_', ' ').title(),
                                'ticker': None
                            }
                            all_news.append(article)
                        except Exception as e:
                            log_error(f"Error parsing market news entry: {str(e)}")
                            continue
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    log_error(f"Error fetching market news from {source_name}: {str(e)}")
                    continue
            
            # Sort by date and return top articles
            sorted_news = sorted(all_news, key=lambda x: x.get('published_date', datetime(1900, 1, 1)), reverse=True)
            return sorted_news[:limit]
            
        except Exception as e:
            log_error(f"Error getting market news: {str(e)}")
            return []

 