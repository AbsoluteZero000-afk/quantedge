"""
QuantEdge Risk Management Module

Implements position sizing algorithms including Kelly Criterion,
fixed fractional, volatility-based sizing, and portfolio risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
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
    
    def calculate_volatility_scaled_size(self, returns: pd.Series,
                                       target_volatility: float = 0.15) -> Tuple[float, Dict[str, float]]:
        """Calculate position size based on volatility scaling."""
        if len(returns) < 10:
            return self.config.default_position_size, {"error": "insufficient_data"}
        
        returns_clean = returns.dropna()
        current_vol = returns_clean.std() * np.sqrt(252)
        
        if current_vol <= 0:
            return self.config.default_position_size, {"error": "zero_volatility"}
        
        vol_scaling_factor = target_volatility / current_vol
        scaled_size = self.config.default_position_size * vol_scaling_factor
        final_size = max(0.005, min(scaled_size, self.config.max_position_size))
        
        risk_metrics = {
            "current_volatility": current_vol,
            "target_volatility": target_volatility,
            "scaling_factor": vol_scaling_factor,
            "final_size": final_size
        }
        
        return final_size, risk_metrics
    
    def calculate_position_size(self, symbol: str, 
                              returns: pd.Series,
                              method: PositionSizingMethod = PositionSizingMethod.KELLY_CRITERION) -> PositionSizeResult:
        """Calculate optimal position size for a given symbol."""
        
        if method == PositionSizingMethod.KELLY_CRITERION:
            recommended_size, risk_metrics = self.calculate_kelly_fraction(returns)
            sizing_method = "kelly_criterion"
            
        elif method == PositionSizingMethod.VOLATILITY_SCALED:
            recommended_size, risk_metrics = self.calculate_volatility_scaled_size(returns)
            sizing_method = "volatility_scaled"
            
        else:
            recommended_size = self.config.default_position_size
            sizing_method = "fixed_fractional"
            risk_metrics = {"base_size": recommended_size}
        
        # Apply additional risk controls
        risk_adjusted_size = min(recommended_size, self.config.max_position_size)
        
        return PositionSizeResult(
            symbol=symbol,
            recommended_size=recommended_size,
            risk_adjusted_size=risk_adjusted_size,
            sizing_method=sizing_method,
            risk_metrics=risk_metrics
        )
    
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
    """Example usage of the risk manager."""
    logger.info("QuantEdge Risk Manager - Example usage")
    
    risk_manager = RiskManager()
    
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.015, 252), index=dates)
    
    # Calculate position size
    result = risk_manager.calculate_position_size('AAPL', returns)
    
    logger.info("Position size calculated",
               symbol=result.symbol,
               recommended_size=f"{result.recommended_size:.2%}",
               risk_adjusted_size=f"{result.risk_adjusted_size:.2%}")


if __name__ == "__main__":
    main()
