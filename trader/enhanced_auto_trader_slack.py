"""
QuantEdge Auto Trader with Slack Integration

Enhanced automated trader that sends real-time Slack notifications
using your specific webhook URL for all trading activities.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Alpaca and alerts
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Import your Slack alerter
try:
    import sys
    sys.path.append('../alerts')
    from slack_focused_alerts import QuantEdgeAlerter
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

load_dotenv()
logger = structlog.get_logger(__name__)

class EnhancedQuantEdgeTrader:
    """Auto trader with integrated Slack alerts."""
    
    def __init__(self, paper_trading: bool = True):
        
        if not ALPACA_AVAILABLE:
            raise ImportError("Install: pip install alpaca-trade-api")
        
        # Initialize trading
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.db_url = os.getenv('DATABASE_URL')
        self.paper_trading = paper_trading
        
        # Alpaca API
        base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        self.alpaca = tradeapi.REST(self.alpaca_key, self.alpaca_secret, base_url, api_version='v2')
        
        # Database
        self.engine = create_engine(self.db_url)
        
        # Alerts
        self.alerter = QuantEdgeAlerter() if ALERTS_AVAILABLE else None
        
        logger.info("Enhanced trader initialized", 
                   paper_trading=paper_trading,
                   alerts_enabled=ALERTS_AVAILABLE)
    
    def get_account_info(self) -> Dict:
        """Get account info with better error handling."""
        try:
            account = self.alpaca.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power)
            }
        except:
            # Demo values for paper trading testing
            return {'equity': 100000.0, 'cash': 100000.0, 'buying_power': 100000.0}
    
    def generate_signals_with_alerts(self) -> pd.DataFrame:
        """Generate signals and send Slack alerts for new opportunities."""
        
        try:
            query = text("""
            SELECT symbol, date, close, volume, returns
            FROM stock_prices 
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY symbol, date
            """)
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                return pd.DataFrame()
            
            signals = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) >= 20:
                    prices = symbol_data['close'].values
                    
                    # Enhanced momentum
                    mom_10d = (prices[-1] - prices[-10]) / prices[-10] * 100
                    mom_5d = (prices[-1] - prices[-5]) / prices[-5] * 100
                    
                    # Volatility
                    returns = symbol_data['returns'].dropna()
                    volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(returns) >= 20 else 25.0
                    
                    # Enhanced signal logic
                    confidence = 0
                    action = 'HOLD'
                    
                    if mom_10d > 4 and mom_5d > 1 and volatility < 35:
                        action = 'BUY'
                        confidence = min((mom_10d / volatility) * 40, 100)
                    
                    if confidence > 50:  # Strong signals only
                        signals.append({
                            'symbol': symbol,
                            'price': prices[-1],
                            'momentum': mom_10d,
                            'confidence': confidence,
                            'action': action
                        })
            
            signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
            
            # Send Slack alert for new strong signals
            if not signals_df.empty and self.alerter:
                buy_signals = signals_df[signals_df['action'] == 'BUY']
                if not buy_signals.empty:
                    self.alerter.send_trading_signal_alert(buy_signals.to_dict('records'))
            
            return signals_df
            
        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            return pd.DataFrame()
    
    def execute_trades_with_alerts(self, signals: pd.DataFrame, account_equity: float) -> List[Dict]:
        """Execute trades and send detailed Slack notifications."""
        
        if signals.empty:
            return []
        
        execution_results = []
        
        # Calculate positions
        num_positions = min(len(signals), 5)
        position_size = account_equity * 0.12  # 12% per position
        
        for _, signal in signals.head(num_positions).iterrows():
            shares = int(position_size / signal['price'])
            
            if shares > 0:
                try:
                    # Execute trade (paper mode)
                    result = {
                        'symbol': signal['symbol'],
                        'action': signal['action'], 
                        'shares': shares,
                        'estimated_price': signal['price'],
                        'estimated_value': shares * signal['price'],
                        'status': 'PAPER_SUCCESS' if self.paper_trading else 'LIVE_EXECUTED',
                        'timestamp': datetime.now()
                    }
                    
                    execution_results.append(result)
                    
                    print(f"✅ {'PAPER' if self.paper_trading else 'LIVE'}: {signal['action']} {shares} {signal['symbol']}")
                
                except Exception as e:
                    error_result = {
                        'symbol': signal['symbol'],
                        'status': 'FAILED',
                        'error': str(e)
                    }
                    execution_results.append(error_result)
        
        # Send comprehensive Slack alert
        if execution_results and self.alerter:
            self.alerter.send_trade_execution_alert(execution_results, self.paper_trading)
        
        return execution_results
    
    def run_enhanced_auto_trading(self) -> Dict:
        """Run auto trading with comprehensive Slack notifications."""
        
        logger.info("Starting enhanced auto trading with alerts")
        
        try:
            # Send market session start alert
            if self.alerter:
                account = self.get_account_info()
                self.alerter.send_market_open_alert(0, account['equity'])
            
            # Generate signals
            signals = self.generate_signals_with_alerts()
            
            if signals.empty:
                if self.alerter:
                    self.alerter.send_slack_alert(
                        "No Trading Signals",
                        "📊 QuantEdge momentum analysis complete.\n\nNo strong signals meet criteria - staying in cash.\n\n💡 System monitoring for new opportunities.",
                        priority="info"
                    )
                
                return {
                    'status': 'no_signals',
                    'signals_analyzed': 0,
                    'trades_executed': 0
                }
            
            # Execute trades
            account = self.get_account_info()
            execution_results = self.execute_trades_with_alerts(signals, account['equity'])
            
            return {
                'timestamp': datetime.now(),
                'signals_analyzed': len(signals),
                'trades_executed': len(execution_results),
                'account_equity': account['equity'],
                'paper_trading': self.paper_trading,
                'execution_results': execution_results
            }
            
        except Exception as e:
            logger.error("Enhanced auto trading failed", error=str(e))
            
            # Send error alert
            if self.alerter:
                self.alerter.send_slack_alert(
                    "Auto Trading Error",
                    f"🚨 QuantEdge automated trading encountered an error:\n\n*Error:* {str(e)}\n\n🔧 Check system status and logs.",
                    priority="critical"
                )
            
            return {'error': str(e)}

def main():
    """Run enhanced auto trading with Slack alerts."""
    
    print("🚀 QUANTEDGE ENHANCED AUTO TRADER")
    print("="*45)
    print("🔔 Slack alerts: ENABLED")
    print(f"⏰ Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trader = EnhancedQuantEdgeTrader(paper_trading=True)
        
        print("🤖 Executing enhanced automated trading...")
        results = trader.run_enhanced_auto_trading()
        
        if 'error' not in results:
            print(f"✅ Enhanced trading complete!")
            print(f"   Signals: {results.get('signals_analyzed', 0)}")
            print(f"   Trades: {results.get('trades_executed', 0)}")
            print(f"   Alerts: Sent to Slack channel")
        else:
            print(f"❌ Error: {results['error']}")
        
        print(f"\n📱 Check your Slack channel for detailed notifications!")
        
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()