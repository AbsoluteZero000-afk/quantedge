"""
Enhanced QuantEdge Backtesting Engine - Updated for Real Data Integration

Vectorized backtesting framework that works with your live database data,
momentum strategies, and provides comprehensive performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Protocol
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logger = structlog.get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005   # 0.05% slippage
    max_positions: int = 10
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
    calmar_ratio: float
    sortino_ratio: float


class Strategy(Protocol):
    """Protocol defining the strategy interface."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market data."""
        ...
    
    def get_name(self) -> str:
        """Return strategy name."""
        ...


class RealDataBacktester:
    """Backtester integrated with your live PostgreSQL data."""
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize the backtester."""
        self.config = config or BacktestConfig()
        self.portfolio_values: pd.Series = pd.Series(dtype=float)
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        self.trades_log: List[Dict] = []
        
        # Database connection
        self.db_url = os.getenv('DATABASE_URL')
        if self.db_url:
            self.engine = create_engine(self.db_url)
        
        logger.info("RealDataBacktester initialized", 
                   initial_capital=self.config.initial_capital)
    
    def load_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load real market data from your database."""
        if not self.db_url:
            logger.error("No database URL configured")
            return {}
        
        market_data = {}
        
        try:
            for symbol in symbols:
                query = text("""
                SELECT date, open, high, low, close, volume, returns
                FROM stock_prices 
                WHERE symbol = :symbol 
                AND date BETWEEN :start_date AND :end_date
                ORDER BY date
                """)
                
                df = pd.read_sql(
                    query, 
                    self.engine, 
                    params={
                        'symbol': symbol,
                        'start_date': start_date.date(),
                        'end_date': end_date.date()
                    }
                )
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    market_data[symbol] = df
                    logger.info("Loaded data for symbol", symbol=symbol, rows=len(df))
                else:
                    logger.warning("No data found for symbol", symbol=symbol)
        
        except Exception as e:
            logger.error("Failed to load market data", error=str(e))
            return {}
        
        return market_data
    
    def _calculate_transaction_costs(self, trade_value: float) -> Tuple[float, float]:
        """Calculate commission and slippage costs."""
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * self.config.slippage_rate
        return commission, slippage
    
    def _execute_trades(self, signals: pd.DataFrame, prices: pd.DataFrame, 
                       initial_capital: float) -> Tuple[pd.Series, List[Dict]]:
        """Execute trades based on signals and return portfolio values."""
        
        # Align dates
        common_dates = signals.index.intersection(prices.index)
        signals = signals.loc[common_dates]
        prices = prices.loc[common_dates]
        
        portfolio_values = []
        trades_log = []
        cash = initial_capital
        positions = pd.Series(0.0, index=signals.columns)
        
        for date in common_dates:
            current_signals = signals.loc[date]
            current_prices = prices.loc[date]
            
            # Calculate current portfolio value
            position_values = positions * current_prices
            total_portfolio_value = cash + position_values.sum()
            portfolio_values.append(total_portfolio_value)
            
            # Check for rebalancing (weekly for now)
            if date.weekday() == 0 or date == common_dates[0]:  # Monday or first day
                # Calculate target positions
                target_weights = current_signals.fillna(0)
                target_positions = (target_weights * total_portfolio_value / current_prices).fillna(0)
                
                # Execute trades
                trades = target_positions - positions
                
                for symbol in trades.index:
                    if abs(trades[symbol]) > 0.01:  # Minimum trade size
                        trade_value = abs(trades[symbol] * current_prices[symbol])
                        commission, slippage = self._calculate_transaction_costs(trade_value)
                        
                        # Update cash and positions
                        cash -= trades[symbol] * current_prices[symbol] + commission + slippage
                        positions[symbol] = target_positions[symbol]
                        
                        # Log trade
                        trades_log.append({
                            'date': date,
                            'symbol': symbol,
                            'shares': trades[symbol],
                            'price': current_prices[symbol],
                            'value': trade_value,
                            'commission': commission,
                            'slippage': slippage
                        })
        
        portfolio_series = pd.Series(portfolio_values, index=common_dates)
        return portfolio_series, trades_log
    
    def _calculate_performance_metrics(self, portfolio_values: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(portfolio_values) < 2:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        trading_days = len(returns)
        years = trading_days / 252
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate / 252
        
        sharpe_ratio = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
        sortino_ratio = (excess_returns.mean() / (downside_std / np.sqrt(252))) if downside_std > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = portfolio_values / portfolio_values.iloc[0]
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate calculation
        win_rate = (returns > 0).mean()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades_log),
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
    
    def run_backtest(self, symbols: List[str], strategy: Strategy, 
                    start_date: datetime = None, end_date: datetime = None) -> Tuple[PerformanceMetrics, pd.Series]:
        """Run complete backtest with real data."""
        
        # Default date range if not provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        logger.info("Starting backtest", 
                   strategy=strategy.get_name(),
                   symbols=symbols,
                   start_date=start_date.date(),
                   end_date=end_date.date())
        
        # Load market data
        market_data = self.load_market_data(symbols, start_date, end_date)
        
        if not market_data:
            logger.error("No market data loaded")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0), pd.Series()
        
        # Combine price data
        prices_df = pd.DataFrame({
            symbol: data['close'] 
            for symbol, data in market_data.items()
        })
        
        # Generate signals
        signals_df = strategy.generate_signals(market_data)
        
        # Execute backtest
        portfolio_values, trades_log = self._execute_trades(
            signals_df, prices_df, self.config.initial_capital
        )
        
        self.portfolio_values = portfolio_values
        self.trades_log = trades_log
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(portfolio_values)
        
        logger.info("Backtest completed",
                   total_return=f"{metrics.total_return:.2%}",
                   sharpe_ratio=f"{metrics.sharpe_ratio:.2f}",
                   max_drawdown=f"{metrics.max_drawdown:.2%}",
                   total_trades=metrics.total_trades)
        
        return metrics, portfolio_values
    
    def get_trades_log(self) -> pd.DataFrame:
        """Return detailed trades log."""
        if self.trades_log:
            return pd.DataFrame(self.trades_log)
        return pd.DataFrame()


class MomentumBacktestStrategy:
    """Momentum strategy specifically for backtesting."""
    
    def __init__(self, lookback_days: int = 20, top_pct: float = 0.6):
        self.lookback_days = lookback_days
        self.top_pct = top_pct
        self.name = f"momentum_{lookback_days}d_top{int(top_pct*100)}pct"
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate momentum-based trading signals."""
        
        if not market_data:
            return pd.DataFrame()
        
        # Get all dates
        all_dates = set()
        for data in market_data.values():
            all_dates.update(data.index)
        
        all_dates = sorted(list(all_dates))
        symbols = list(market_data.keys())
        
        # Initialize signals dataframe
        signals = pd.DataFrame(0.0, index=all_dates, columns=symbols)
        
        for date in all_dates:
            momentum_scores = {}
            
            # Calculate momentum for each symbol
            for symbol in symbols:
                data = market_data[symbol]
                
                # Get data up to current date
                historical_data = data[data.index <= date]
                
                if len(historical_data) >= self.lookback_days:
                    prices = historical_data['close'].values
                    
                    # Calculate momentum
                    momentum = (prices[-1] - prices[-self.lookback_days]) / prices[-self.lookback_days]
                    
                    # Calculate volatility for risk adjustment
                    returns = historical_data['returns'].dropna()
                    if len(returns) >= 10:
                        volatility = returns.rolling(20).std().iloc[-1]
                        risk_adj_momentum = momentum / volatility if volatility > 0 else 0
                    else:
                        risk_adj_momentum = momentum
                    
                    momentum_scores[symbol] = risk_adj_momentum
            
            # Select top performers
            if momentum_scores:
                sorted_scores = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                top_count = max(1, int(len(sorted_scores) * self.top_pct))
                
                # Equal weight allocation to top performers
                weight = 1.0 / top_count
                
                for i in range(top_count):
                    symbol = sorted_scores[i][0]
                    if sorted_scores[i][1] > 0:  # Only positive momentum
                        signals.loc[date, symbol] = weight
        
        return signals
    
    def get_name(self) -> str:
        return self.name


def main():
    """Example usage of the enhanced backtester."""
    logger.info("QuantEdge Real Data Backtester - Example")
    
    # Your current symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Create backtester and strategy
    backtester = RealDataBacktester()
    strategy = MomentumBacktestStrategy(lookback_days=10, top_pct=0.6)
    
    # Run backtest
    start_date = datetime.now() - timedelta(days=90)  # 3 months
    end_date = datetime.now()
    
    metrics, portfolio_values = backtester.run_backtest(
        symbols, strategy, start_date, end_date
    )
    
    print(f"\n🎯 BACKTEST RESULTS - {strategy.get_name()}")
    print("="*50)
    print(f"📊 Total Return: {metrics.total_return:.2%}")
    print(f"📈 Annualized Return: {metrics.annualized_return:.2%}")
    print(f"🌡️ Volatility: {metrics.volatility:.2%}")
    print(f"⚡ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"📉 Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"🎯 Win Rate: {metrics.win_rate:.2%}")
    print(f"💼 Total Trades: {metrics.total_trades}")
    print(f"🏆 Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"📊 Sortino Ratio: {metrics.sortino_ratio:.2f}")
    
    # Show trades log
    trades_df = backtester.get_trades_log()
    if not trades_df.empty:
        print(f"\n📝 Recent Trades:")
        print(trades_df.tail().to_string())


if __name__ == "__main__":
    main()