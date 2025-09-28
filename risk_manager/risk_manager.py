"""
QuantEdge Risk Management Module

Implements position sizing algorithms including Kelly Criterion,
fixed fractional, volatility-based sizing, and portfolio risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class RiskConfig:
    """Risk management configuration parameters."""
    max_position_size: float = 0.10        # 10% max per position
    max_portfolio_risk: float = 0.20       # 20% max portfolio risk
    default_position_size: float = 0.02    # 2% default risk per trade
    kelly_fraction: float = 0.25           # Quarter Kelly for safety
    volatility_lookback: int = 20          # Days for volatility calculation


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    symbol: str
    recommended_size: float
    risk_adjusted_size: float
    sizing_method: str
    risk_metrics: Dict[str, float]


class RiskManager:
    """
    Comprehensive risk management system with multiple position sizing
    algorithms and portfolio-level risk controls.
    """
    
    def __init__(self, config: RiskConfig = None):
        """Initialize the risk manager."""
        self.config = config or RiskConfig()
        self.position_history: Dict[str, List[float]] = {}
        
        logger.info("RiskManager initialized", config=self.config)
    
    def calculate_kelly_fraction(self, returns: pd.Series, 
                                min_observations: int = 30) -> Tuple[float, Dict[str, float]]:
        """Calculate Kelly Criterion fraction for position sizing."""
        if len(returns) < min_observations:
            logger.warning("Insufficient data for Kelly calculation",
                         observations=len(returns), required=min_observations)
            return 0.0, {"error": "insufficient_data"}
        
        # Remove NaN values and outliers
        clean_returns = returns.dropna()
        q99 = clean_returns.quantile(0.99)
        q01 = clean_returns.quantile(0.01)
        clean_returns = clean_returns[(clean_returns >= q01) & (clean_returns <= q99)]
        
        if len(clean_returns) < min_observations:
            return 0.0, {"error": "insufficient_clean_data"}
        
        # Calculate Kelly fraction
        mean_return = clean_returns.mean()
        variance = clean_returns.var()
        risk_free_rate = 0.02 / 252
        
        if variance <= 0:
            return 0.0, {"error": "zero_variance"}
        
        kelly_fraction = (mean_return - risk_free_rate) / variance
        adjusted_kelly = kelly_fraction * self.config.kelly_fraction
        
        # Safety checks
        if kelly_fraction < 0:
            adjusted_kelly = 0.0
        elif kelly_fraction > 2.0:
            adjusted_kelly = self.config.max_position_size
        
        risk_metrics = {
            "raw_kelly": kelly_fraction,
            "adjusted_kelly": adjusted_kelly,
            "mean_return": mean_return,
            "variance": variance,
            "sharpe_ratio": mean_return / np.sqrt(variance) if variance > 0 else 0,
            "win_rate": (clean_returns > 0).mean()
        }
        
        return max(0.0, min(adjusted_kelly, self.config.max_position_size)), risk_metrics
    
    def calculate_position_size_for_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate position sizes for your momentum trading signals."""
        print("⚖️ QUANTEDGE RISK MANAGEMENT")
        print("="*50)
        
        results = {}
        
        for symbol, signal_data in signals.items():
            momentum = signal_data.get('momentum', 0)
            allocation = signal_data.get('allocation', 0.02)
            
            # Risk-based position sizing
            base_size = self.config.default_position_size
            
            # Adjust for momentum strength
            momentum_multiplier = min(1.5, 1 + (momentum / 100))
            adjusted_size = base_size * momentum_multiplier
            
            # Apply maximum position limits
            final_size = min(adjusted_size, self.config.max_position_size)
            
            results[symbol] = {
                'recommended_size': final_size,
                'base_allocation': allocation,
                'momentum': momentum,
                'risk_adjusted': True
            }
            
            print(f"📊 {symbol}:")
            print(f"   🎯 Momentum: +{momentum:.2f}%")
            print(f"   💰 Base Size: {base_size:.1%}")
            print(f"   📈 Momentum Adj: {momentum_multiplier:.2f}x")
            print(f"   ✅ Final Size: {final_size:.1%}")
            print()
        
        return results
    
    def check_portfolio_risk(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Check overall portfolio risk metrics and constraints."""
        total_risk = sum(positions.values())
        
        risk_assessment = {
            "total_position_risk": total_risk,
            "risk_budget_used": total_risk / self.config.max_portfolio_risk,
            "number_of_positions": len(positions),
            "largest_position": max(positions.values()) if positions else 0,
            "violations": []
        }
        
        # Check violations
        if total_risk > self.config.max_portfolio_risk:
            risk_assessment["violations"].append(
                f"Total portfolio risk ({total_risk:.1%}) exceeds limit"
            )
        
        if max(positions.values()) > self.config.max_position_size:
            risk_assessment["violations"].append(
                f"Individual position exceeds limit"
            )
        
        return risk_assessment


def main():
    """Test risk manager with your momentum signals."""
    print("🚀 QuantEdge Risk Manager - Testing with Your Signals")
    
    risk_manager = RiskManager()
    
    # Your actual momentum signals
    signals = {
        'AAPL': {'momentum': 7.93, 'allocation': 0.50},
        'TSLA': {'momentum': 7.40, 'allocation': 0.50}
    }
    
    # Calculate risk-adjusted position sizes
    risk_adjusted_positions = risk_manager.calculate_position_size_for_signals(signals)
    
    # Check portfolio risk
    position_sizes = {symbol: data['recommended_size'] 
                     for symbol, data in risk_adjusted_positions.items()}
    
    risk_check = risk_manager.check_portfolio_risk(position_sizes)
    
    print("🛡️ PORTFOLIO RISK ASSESSMENT:")
    print(f"   📊 Total Risk: {risk_check['total_position_risk']:.1%}")
    print(f"   📈 Risk Budget Used: {risk_check['risk_budget_used']:.1%}")
    print(f"   🔢 Number of Positions: {risk_check['number_of_positions']}")
    
    if risk_check['violations']:
        print("⚠️ RISK VIOLATIONS:")
        for violation in risk_check['violations']:
            print(f"   ❌ {violation}")
    else:
        print("✅ No risk violations - portfolio is properly sized!")


if __name__ == "__main__":
    main()
