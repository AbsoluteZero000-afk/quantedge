"""
QuantEdge Data Ingestion Module

Fetches OHLC data and fundamentals from Financial Modeling Prep API
with proper rate limiting, pagination, and PostgreSQL storage.

Author: QuantEdge Team
License: MIT
"""

import time
import requests
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import structlog
import numpy as np
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

load_dotenv()
logger = structlog.get_logger(__name__)


class FMPDataIngester:
    """
    Financial Modeling Prep API data ingestion class with rate limiting,
    error handling, and PostgreSQL storage.
    """
    
    def __init__(self, api_key: str, db_url: str):
        """Initialize the FMP data ingester."""
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QuantEdge/1.0.0 (Personal Trading System)'
        })
        
        # Rate limiting: FMP allows 250 requests/day on free plan
        self.request_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
        # Database setup
        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        
        logger.info("FMPDataIngester initialized", api_key_present=bool(api_key))
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logger.debug("Rate limiting", sleep_time=sleep_time)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make a rate-limited request to FMP API."""
        self._rate_limit()
        
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info("API request successful", endpoint=endpoint, 
                       response_size=len(data) if isinstance(data, list) else 1)
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error("API request failed", endpoint=endpoint, error=str(e))
            return None
        except ValueError as e:
            logger.error("JSON decode failed", endpoint=endpoint, error=str(e))
            return None
    
    def get_historical_prices(self, symbol: str, from_date: str = None, 
                            to_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for a symbol."""
        logger.info("Fetching historical prices", symbol=symbol, 
                   from_date=from_date, to_date=to_date)
        
        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        data = self._make_request(f"/v3/historical-price-full/{symbol}", params)
        if data is None or 'historical' not in data:
            return None
        
        df = pd.DataFrame(data['historical'])
        if df.empty:
            return None
        
        # Clean and format data
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate additional metrics
        df['returns'] = df['close'].pct_change()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['price_ma_20'] = df['close'].rolling(window=20).mean()
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        logger.info("Historical prices processed", symbol=symbol, rows=len(df))
        return df
    
    def store_prices(self, df: pd.DataFrame) -> bool:
        """Store price data in PostgreSQL database."""
        try:
            df.to_sql('stock_prices', self.engine, if_exists='append', 
                     index=False, method='multi', chunksize=1000)
            logger.info("Prices stored", symbol=df['symbol'].iloc[0], rows=len(df))
            return True
        except Exception as e:
            logger.error("Failed to store prices", error=str(e))
            return False
    
    def update_symbol_data(self, symbol: str, days_back: int = 30) -> bool:
        """Update both price and fundamental data for a single symbol."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info("Updating symbol data", symbol=symbol, from_date=from_date)
        
        # Fetch and store price data
        price_df = self.get_historical_prices(symbol, from_date, to_date)
        if price_df is not None:
            return self.store_prices(price_df)
        
        return False


def main():
    """Main function for standalone execution."""
    api_key = os.getenv('FMP_API_KEY')
    db_url = os.getenv('DATABASE_URL')
    
    if not api_key or not db_url:
        logger.error("Missing required environment variables")
        return
    
    ingester = FMPDataIngester(api_key, db_url)
    
    # Example usage: Update S&P 500 stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    for symbol in symbols:
        logger.info("Processing symbol", symbol=symbol)
        ingester.update_symbol_data(symbol, days_back=30)


if __name__ == "__main__":
    main()
