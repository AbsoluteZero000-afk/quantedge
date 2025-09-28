"""
QuantEdge Daily Trading Runner - FIXED VERSION

Fixed type hints and enhanced error handling for daily automation.
"""

import os
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import structlog
from typing import Dict, List  # Fixed import

load_dotenv()
logger = structlog.get_logger(__name__)

class DailyTradingRunner:
    """Fixed daily trading automation."""
    
    def __init__(self):
        self.paper_trading = True
        self.max_daily_trades = 5
        self.trading_log_file = 'logs/daily_trading.json'
        
        os.makedirs('../logs', exist_ok=True)
    
    def is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        today = datetime.now()
        
        # Skip weekends
        if today.weekday() >= 5:
            return False
        
        return True
    
    def run_daily_trading(self) -> Dict:
        """Execute daily trading with safety checks."""
        
        print("🤖 QUANTEDGE DAILY TRADING")
        print("="*40)
        
        if not self.is_trading_day():
            print("📅 Weekend - skipping trading")
            return {'status': 'skipped', 'reason': 'weekend'}
        
        try:
            # Import and run auto trader
            import sys
            sys.path.append('')
            from fixed_auto_trader import QuantEdgeAutoTrader
            
            trader = QuantEdgeAutoTrader(paper_trading=self.paper_trading)
            
            print("🔄 Running automated momentum strategy...")
            results = trader.safe_auto_trade(max_trades_per_day=self.max_daily_trades)
            
            if 'error' not in results:
                print(f"✅ Trading complete - {results.get('trades_executed', 0)} trades")
            else:
                print(f"❌ Trading failed: {results['error']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Daily trading error: {e}")
            return {'error': str(e)}

def main():
    """Run daily trading."""
    runner = DailyTradingRunner()
    results = runner.run_daily_trading()
    
    if results.get('status') == 'skipped':
        print("⏭️ Trading skipped")
    elif 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print("✅ Daily trading completed!")

if __name__ == "__main__":
    main()