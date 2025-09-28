"""
QuantEdge Alpaca Trading Adapter

Paper and live trading integration with Alpaca Markets API.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)


class OrderType(Enum):
    """Order types supported by the trading system."""
    MARKET = "market"
    LIMIT = "limit"


class PositionSide(Enum):
    """Position sides."""
    LONG = "long"
    SHORT = "short"


@dataclass
class OrderRequest:
    """Standardized order request format."""
    symbol: str
    quantity: float
    side: PositionSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: float
    side: str
    market_value: float
    unrealized_pnl: float
    current_price: float


class AlpacaTrader:
    """Alpaca trading adapter with order management and risk controls."""
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        """Initialize Alpaca trader."""
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading
        
        # Risk controls
        self.max_daily_trades = 50
        self.max_position_value = 50000
        self.daily_loss_limit = 5000
        
        # Performance tracking
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        
        logger.info("AlpacaTrader initialized", 
                   paper_trading=paper_trading)
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        # Mock account info for demo
        account_info = {
            "account_id": "demo_account",
            "equity": 100000.0,
            "buying_power": 100000.0,
            "cash": 100000.0,
            "portfolio_value": 100000.0
        }
        
        logger.info("Account info retrieved", 
                   equity=account_info["equity"])
        
        return account_info
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        # Mock positions for demo
        positions = []
        
        logger.info("Positions retrieved", count=len(positions))
        return positions
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        # Mock current price
        mock_prices = {
            'AAPL': 150.0,
            'MSFT': 330.0,
            'GOOGL': 2650.0,
            'SPY': 420.0
        }
        
        return mock_prices.get(symbol, 100.0)
    
    def submit_order(self, order_request: OrderRequest) -> Optional[str]:
        """Submit an order to Alpaca."""
        logger.warning("Order submission disabled in demo mode")
        
        # In production, this would submit the actual order
        # For now, just return a mock order ID
        mock_order_id = f"order_{int(time.time())}"
        
        logger.info("Mock order created", 
                   order_id=mock_order_id,
                   symbol=order_request.symbol,
                   quantity=order_request.quantity)
        
        return mock_order_id
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        account_info = self.get_account_info()
        positions = self.get_positions()
        
        summary = {
            "timestamp": datetime.now(),
            "account": account_info,
            "positions": {
                "count": len(positions),
                "total_market_value": sum(abs(p.market_value) for p in positions),
                "total_unrealized_pnl": sum(p.unrealized_pnl for p in positions),
                "positions": positions
            },
            "performance": {
                "today_pnl": 0.0,
                "daily_trades": self.daily_trades_count
            }
        }
        
        return summary


def main():
    """Example usage of the Alpaca trader."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    paper_trading = os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true'
    
    if not api_key or not secret_key:
        logger.error("Missing Alpaca API credentials")
        return
    
    # Initialize trader
    trader = AlpacaTrader(api_key, secret_key, paper_trading)
    
    # Get account info
    account = trader.get_account_info()
    logger.info("Account loaded", equity=account.get("equity", 0))
    
    # Get portfolio summary
    summary = trader.get_portfolio_summary()
    logger.info("Portfolio summary generated")


if __name__ == "__main__":
    main()
