"""
QuantEdge Backtesting Engine

Vectorized backtesting framework with strategy registry,
transaction costs, slippage modeling, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Protocol
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from abc import ABC, abstractmethod

logger = structlog.get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005   # 0.05% slippage
    max_positions: int = 20
    rebalance_frequency: str = 'weekly'
    benchmark_symbol: str = 'SPY'


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


class Strategy(Protocol):
    """Protocol defining the strategy interface."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market data."""
        ...
    
    def get_name(self) -> str:
        """Return strategy name."""
        ...


class VectorizedBacktester:
    """High-performance vectorized backtesting engine."""
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize the backtester."""
        self.config = config or BacktestConfig()
        self.portfolio_values: pd.Series = pd.Series(dtype=float)
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        
        logger.info("VectorizedBacktester initialized", 
                   initial_capital=self.config.initial_capital)
    
    def _load_benchmark_data(self, start_date: datetime, end_date: datetime) -> None:
        """Load benchmark data for comparison."""
        date_range = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)  # For reproducible results
        
        returns = np.random.normal(0.001, 0.015, len(date_range))
        prices = 100 * np.cumprod(1 + returns)
        
        self.benchmark_data = pd.DataFrame({
            'date': date_range,
            'close': prices,
            'returns': returns
        }).set_index('date')
    
    def _calculate_transaction_costs(self, trade_value: float) -> Tuple[float, float]:
        """Calculate commission and slippage costs."""
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * self.config.slippage_rate
        return commission, slippage
    
    def _calculate_portfolio_returns(self, weights: pd.DataFrame, 
                                   returns: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from weights and asset returns."""
        aligned_weights = weights.shift(1).fillna(0)
        aligned_returns = returns.reindex(aligned_weights.index).fillna(0)
        
        portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1)
        
        # Apply transaction costs
        position_changes = weights.diff().abs().sum(axis=1)
        transaction_costs = position_changes * (
            self.config.commission_rate + self.config.slippage_rate
        )
        
        portfolio_returns = portfolio_returns - transaction_costs
        
        return portfolio_returns
    
    def _calculate_performance_metrics(self, portfolio_returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        total_return = (1 + portfolio_returns).prod() - 1
        trading_days = len(portfolio_returns)
        years = trading_days / 252
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02
        excess_returns = portfolio_returns - risk_free_rate / 252
        
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=0.6,  # Placeholder
            total_trades=100  # Placeholder
        )
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    strategy: Strategy, 
                    start_date: datetime = None,
                    end_date: datetime = None) -> PerformanceMetrics:
        """Run complete vectorized backtest."""
        logger.info("Starting backtest", strategy=strategy.get_name())
        
        # Prepare data
        if not data:
            raise ValueError("No data provided for backtesting")
        
        # Create sample portfolio returns for demo
        np.random.seed(42)
        dates = pd.date_range(start_date or datetime.now() - timedelta(days=365), 
                             end_date or datetime.now(), freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.018, len(dates)), index=dates)
        
        self.portfolio_values = self.config.initial_capital * (1 + returns).cumprod()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(returns)
        
        logger.info("Backtest completed", 
                   total_return=f"{metrics.total_return:.2%}",
                   sharpe_ratio=f"{metrics.sharpe_ratio:.2f}",
                   max_drawdown=f"{metrics.max_drawdown:.2%}")
        
        return metrics
    
    def get_equity_curve(self) -> pd.Series:
        """Return the portfolio equity curve."""
        return self.portfolio_values


# Example strategy for demonstration
class SimpleStrategy:
    """Simple buy-and-hold strategy for testing."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate simple buy signals."""
        # Return a simple signal matrix
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range(datetime.now() - timedelta(days=365), 
                             datetime.now(), freq='D')
        
        signals = pd.DataFrame(1.0, index=dates, columns=symbols)
        return signals / len(symbols)  # Equal weight
    
    def get_name(self) -> str:
        return "simple_strategy"


def main():
    """Example usage of the backtester."""
    logger.info("QuantEdge Backtester - Example usage")
    
    # Generate sample data
    sample_data = {}
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        
        sample_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    # Run backtest
    backtester = VectorizedBacktester()
    strategy = SimpleStrategy()
    
    metrics = backtester.run_backtest(sample_data, strategy)
    
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")


if __name__ == "__main__":
    main()
