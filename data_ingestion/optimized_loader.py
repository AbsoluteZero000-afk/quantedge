"""
Optimized Data Loader for 10-Stock QuantEdge System

Loads and maintains data for your core 10-stock universe with proper
API management, database handling, and Alpaca compatibility.
"""

import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text
import structlog
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
logger = structlog.get_logger(__name__)

class OptimizedDataLoader:
    """Optimized data loader for your 10-stock trading universe."""
    
    def __init__(self, api_key: str, db_url: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api"
        self.session = requests.Session()
        self.request_delay = 1.2  # Conservative rate limiting
        self.last_request_time = 0
        
        self.engine = create_engine(db_url)
        logger.info("OptimizedDataLoader initialized")
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: dict = None):
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
            
            # Log API limit info if available
            remaining = response.headers.get('X-Ratelimit-Remaining')
            if remaining:
                logger.info("API request successful", 
                           endpoint=endpoint, 
                           remaining_calls=remaining)
            else:
                logger.info("API request successful", endpoint=endpoint)
            
            return data
        except Exception as e:
            logger.error("API request failed", endpoint=endpoint, error=str(e))
            return None
    
    def get_historical_prices(self, symbol: str, from_date: str = None, to_date: str = None):
        """Fetch comprehensive historical data for a symbol."""
        logger.info("Fetching historical prices", symbol=symbol)
        
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
        
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate comprehensive technical indicators
        self._calculate_technical_indicators(df)
        
        logger.info("Historical prices processed", symbol=symbol, rows=len(df))
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame):
        """Calculate comprehensive technical indicators."""
        
        # Basic returns
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        df['price_ma_20'] = df['close'].rolling(window=20).mean()
        df['price_ma_50'] = df['close'].rolling(window=50).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volatility (annualized)
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_upper'] = df['price_ma_20'] + (2 * df['close'].rolling(20).std())
        df['bb_lower'] = df['price_ma_20'] - (2 * df['close'].rolling(20).std())
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Additional momentum indicators
        df['momentum_10'] = df['close'].pct_change(periods=10) * 100
        df['momentum_20'] = df['close'].pct_change(periods=20) * 100
        
        logger.info("Technical indicators calculated", indicators=9)
    
    def store_prices(self, df: pd.DataFrame):
        """Store price data in PostgreSQL with proper error handling."""
        try:
            symbol = df['symbol'].iloc[0]
            
            # Use proper SQL with text() wrapper
            with self.engine.connect() as conn:
                # Delete existing data for this symbol
                delete_query = text("DELETE FROM stock_prices WHERE symbol = :symbol")
                conn.execute(delete_query, {"symbol": symbol})
                conn.commit()
                
                logger.info("Existing data cleared", symbol=symbol)
            
            # Store new data with error handling for missing columns
            try:
                df.to_sql('stock_prices', self.engine, if_exists='append', 
                         index=False, method='multi', chunksize=1000)
                logger.info("Prices stored successfully", symbol=symbol, rows=len(df))
                return True
                
            except Exception as store_error:
                logger.warning("Full data store failed, trying essential columns only", 
                             error=str(store_error))
                
                # Fallback: store only essential columns
                essential_columns = [
                    'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 
                    'returns', 'price_ma_20', 'volatility_20', 'rsi'
                ]
                
                essential_df = df[essential_columns].copy()
                essential_df.to_sql('stock_prices', self.engine, if_exists='append', 
                                  index=False, method='multi', chunksize=1000)
                
                logger.info("Essential data stored", symbol=symbol, rows=len(essential_df))
                return True
            
        except Exception as e:
            logger.error("Failed to store prices", symbol=symbol, error=str(e))
            return False
    
    def get_core_universe(self):
        """Get your core 10-stock universe optimized for learning and development."""
        
        # Your proven 10-stock universe
        core_universe = {
            # MEGA CAP TECH (Your current winners)
            'AAPL': {'sector': 'Technology', 'name': 'Apple Inc.'},
            'MSFT': {'sector': 'Technology', 'name': 'Microsoft Corporation'},
            'GOOGL': {'sector': 'Technology', 'name': 'Alphabet Inc.'},
            'AMZN': {'sector': 'Technology', 'name': 'Amazon.com Inc.'},
            'TSLA': {'sector': 'Technology', 'name': 'Tesla Inc.'},
            'NVDA': {'sector': 'Technology', 'name': 'NVIDIA Corporation'},
            'META': {'sector': 'Technology', 'name': 'Meta Platforms Inc.'},
            
            # DIVERSIFICATION
            'JPM': {'sector': 'Financial', 'name': 'JPMorgan Chase & Co.'},
            'SPY': {'sector': 'ETF', 'name': 'SPDR S&P 500 ETF'},
            'QQQ': {'sector': 'ETF', 'name': 'Invesco QQQ Trust'}
        }
        
        return core_universe
    
    def load_core_data(self, months_back: int = 12):
        """Load comprehensive data for your core universe."""
        
        universe = self.get_core_universe()
        
        print(f"🚀 LOADING QUANTEDGE CORE DATA")
        print(f"📊 Universe: {len(universe)} symbols")
        print(f"📅 Time period: {months_back} months")
        print("="*50)
        
        # Date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=months_back*30)).strftime('%Y-%m-%d')
        
        successful_loads = 0
        failed_loads = 0
        total_data_points = 0
        
        for i, (symbol, info) in enumerate(universe.items(), 1):
            print(f"📈 [{i:2d}/{len(universe)}] {symbol} - {info['name']}")
            
            try:
                df = self.get_historical_prices(symbol, start_date, end_date)
                
                if df is not None and len(df) > 50:
                    if self.store_prices(df):
                        successful_loads += 1
                        total_data_points += len(df)
                        print(f"   ✅ {len(df)} data points loaded ({info['sector']})")
                    else:
                        failed_loads += 1
                        print(f"   ❌ Database storage failed")
                else:
                    failed_loads += 1
                    print(f"   ❌ Insufficient API data returned")
                    
            except Exception as e:
                failed_loads += 1
                print(f"   ❌ Error: {str(e)}")
        
        print(f"\n🎉 CORE DATA LOADING COMPLETE!")
        print("="*40)
        print(f"✅ Successfully loaded: {successful_loads}/{len(universe)} symbols")
        print(f"📊 Total data points: {total_data_points:,}")
        print(f"📈 Average per symbol: {total_data_points//max(successful_loads,1):,} points")
        print(f"🎯 Success rate: {successful_loads/len(universe)*100:.1f}%")
        
        if successful_loads >= 8:  # At least 80% success
            print(f"\n🎉 EXCELLENT! Your system is ready for:")
            print(f"   • Live momentum signal generation")
            print(f"   • Comprehensive backtesting")
            print(f"   • Risk-managed position sizing")
            print(f"   • Paper trading with Alpaca")
        elif successful_loads >= 5:
            print(f"\n👍 GOOD! Your system can operate with current data")
            print(f"   • Basic momentum signals available")
            print(f"   • Limited backtesting possible")
            print(f"   • Consider retrying failed symbols")
        else:
            print(f"\n⚠️ LIMITED DATA - system needs more symbols")
            print(f"   • Check API key and database connection")
            print(f"   • Retry loading or increase API limits")
        
        return successful_loads, failed_loads, total_data_points
    
    def refresh_daily_data(self):
        """Refresh data for daily trading (uses fewer API calls)."""
        
        universe = self.get_core_universe()
        
        print(f"🔄 DAILY DATA REFRESH")
        print(f"📊 Updating {len(universe)} symbols")
        print("="*30)
        
        # Get last 5 days of data to ensure we have latest
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        updated = 0
        failed = 0
        
        for symbol in universe.keys():
            try:
                print(f"🔄 Updating {symbol}...")
                df = self.get_historical_prices(symbol, start_date, end_date)
                
                if df is not None and len(df) > 0:
                    # For daily updates, append new data only
                    with self.engine.connect() as conn:
                        # Remove last 5 days to avoid duplicates
                        delete_query = text("""
                        DELETE FROM stock_prices 
                        WHERE symbol = :symbol 
                        AND date >= :start_date
                        """)
                        conn.execute(delete_query, {"symbol": symbol, "start_date": start_date})
                        conn.commit()
                    
                    # Add new data
                    df.to_sql('stock_prices', self.engine, if_exists='append', 
                             index=False, method='multi')
                    
                    updated += 1
                    print(f"   ✅ {len(df)} new records")
                else:
                    failed += 1
                    print(f"   ❌ No new data")
                    
            except Exception as e:
                failed += 1
                print(f"   ❌ Error: {str(e)}")
        
        print(f"\n✅ Daily refresh complete!")
        print(f"   Updated: {updated} symbols")
        print(f"   Failed: {failed} symbols")
        
        return updated, failed

def main():
    """Load or refresh your core trading universe."""
    api_key = os.getenv('FMP_API_KEY')
    db_url = os.getenv('DATABASE_URL')
    
    if not api_key:
        print("❌ FMP_API_KEY not found in .env file")
        print("   Get your free key at: https://financialmodelingprep.com")
        return
    
    if not db_url:
        print("❌ DATABASE_URL not found in .env file")
        print("   Make sure PostgreSQL is running: docker-compose up -d postgres")
        return
    
    loader = OptimizedDataLoader(api_key, db_url)
    
    # Check what data already exists
    try:
        engine = create_engine(db_url)
        query = text("SELECT COUNT(DISTINCT symbol) as symbols, MAX(date) as latest_date FROM stock_prices")
        existing_data = pd.read_sql(query, engine)
        
        existing_symbols = existing_data.iloc[0]['symbols']
        latest_date = existing_data.iloc[0]['latest_date']
        
        print(f"📊 CURRENT DATA STATUS:")
        print(f"   Symbols in database: {existing_symbols}")
        if latest_date:
            print(f"   Latest data: {latest_date}")
            days_old = (datetime.now().date() - latest_date).days
            print(f"   Data age: {days_old} days")
        
        # Decide on full load vs refresh
        if existing_symbols < 8 or (latest_date and (datetime.now().date() - latest_date).days > 7):
            print(f"\n🚀 Performing FULL DATA LOAD...")
            loader.load_core_data(months_back=12)
        else:
            print(f"\n🔄 Performing DAILY REFRESH...")
            loader.refresh_daily_data()
            
    except Exception as e:
        print(f"❌ Error checking existing data: {e}")
        print(f"🚀 Performing FULL DATA LOAD...")
        loader.load_core_data(months_back=12)

if __name__ == "__main__":
    main()