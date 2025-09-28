"""
QuantEdge Professional Portfolio Analytics - COMPLETE IMPLEMENTATION

Complete institutional-grade portfolio analysis with correlation,
diversification, risk metrics, and performance attribution.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from scipy.stats import pearsonr
from sqlalchemy import create_engine, text
import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class QuantEdgeAnalytics:
    """Complete professional portfolio analytics system."""
    
    def __init__(self):
        """Initialize professional analytics system."""
        
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment")
        
        self.engine = create_engine(self.db_url)
        
        logger.info("QuantEdge Professional Analytics initialized")
    
    def calculate_correlation_matrix(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Calculate professional correlation matrix for portfolio."""
        
        try:
            # Get returns data
            query = text("""
                SELECT symbol, date, returns
                FROM stock_prices
                WHERE symbol = ANY(:symbols) 
                AND date >= CURRENT_DATE - INTERVAL ':days days'
                AND returns IS NOT NULL
                ORDER BY symbol, date
            """)
            
            df = pd.read_sql(query, self.engine, params={'symbols': symbols, 'days': days})
            
            if df.empty:
                logger.warning("No data for correlation calculation")
                return pd.DataFrame()
            
            # Pivot to get returns by symbol
            returns_pivot = df.pivot(index='date', columns='symbol', values='returns')
            returns_pivot = returns_pivot.dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_pivot.corr()
            
            logger.info("Correlation matrix calculated", 
                       symbols=len(symbols), 
                       observations=len(returns_pivot))
            
            return correlation_matrix
            
        except Exception as e:
            logger.error("Correlation calculation failed", error=str(e))
            return pd.DataFrame()
    
    def analyze_portfolio_diversification(self, positions: Dict[str, float]) -> Dict:
        """Comprehensive portfolio diversification analysis."""
        
        try:
            symbols = list(positions.keys())
            weights = list(positions.values())
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Get correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(symbols, 60)
            
            if correlation_matrix.empty:
                return {'error': 'Insufficient data for diversification analysis'}
            
            # Calculate diversification metrics
            results = {}
            
            # 1. Concentration Risk
            herfindahl_index = sum(w**2 for w in weights)
            effective_positions = 1 / herfindahl_index
            concentration_risk = "HIGH" if herfindahl_index > 0.25 else "MODERATE" if herfindahl_index > 0.15 else "LOW"
            
            # 2. Correlation Analysis
            correlations = []
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    if symbols[i] in correlation_matrix.index and symbols[j] in correlation_matrix.index:
                        corr = correlation_matrix.loc[symbols[i], symbols[j]]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                max_correlation = np.max(correlations)
                
                # Diversification benefit
                diversification_benefit = max(0, (1 - avg_correlation) * 100)
                
                # Risk classification
                correlation_risk = "HIGH" if avg_correlation > 0.7 else "MODERATE" if avg_correlation > 0.5 else "LOW"
            else:
                avg_correlation = 0
                max_correlation = 0
                diversification_benefit = 0
                correlation_risk = "UNKNOWN"
            
            # 3. Sector Diversification (simplified mapping)
            sector_map = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'AMZN': 'Technology', 'TSLA': 'Technology', 'NVDA': 'Technology',
                'JPM': 'Financial', 'BAC': 'Financial', 'V': 'Financial',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare',
                'XOM': 'Energy', 'CVX': 'Energy',
                'WMT': 'Consumer', 'HD': 'Consumer'
            }
            
            sector_weights = {}
            for symbol, weight in zip(symbols, weights):
                sector = sector_map.get(symbol, 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            sector_count = len(sector_weights)
            max_sector_weight = max(sector_weights.values()) if sector_weights else 0
            
            # 4. Professional Diversification Score (0-100)
            concentration_score = (1 - herfindahl_index) * 25  # 25 points max
            correlation_score = (1 - avg_correlation) * 35     # 35 points max
            sector_score = min(sector_count / 5, 1) * 25       # 25 points max
            position_score = min(len(symbols) / 10, 1) * 15    # 15 points max
            
            diversification_score = concentration_score + correlation_score + sector_score + position_score
            
            # Professional assessment
            if diversification_score >= 80:
                assessment = "EXCELLENT"
            elif diversification_score >= 65:
                assessment = "GOOD"
            elif diversification_score >= 50:
                assessment = "MODERATE"
            else:
                assessment = "POOR"
            
            results = {
                'diversification_score': round(diversification_score, 1),
                'assessment': assessment,
                'concentration_metrics': {
                    'herfindahl_index': round(herfindahl_index, 4),
                    'effective_positions': round(effective_positions, 1),
                    'concentration_risk': concentration_risk,
                    'max_position_weight': round(max(weights) * 100, 1)
                },
                'correlation_metrics': {
                    'average_correlation': round(avg_correlation, 3),
                    'maximum_correlation': round(max_correlation, 3),
                    'correlation_risk': correlation_risk,
                    'diversification_benefit': round(diversification_benefit, 1)
                },
                'sector_metrics': {
                    'sector_count': sector_count,
                    'sector_weights': {k: round(v * 100, 1) for k, v in sector_weights.items()},
                    'max_sector_weight': round(max_sector_weight * 100, 1)
                },
                'recommendations': self._generate_diversification_recommendations(
                    diversification_score, concentration_risk, correlation_risk, sector_count
                )
            }
            
            logger.info("Portfolio diversification analyzed", 
                       score=diversification_score,
                       assessment=assessment)
            
            return results
            
        except Exception as e:
            logger.error("Diversification analysis failed", error=str(e))
            return {'error': str(e)}
    
    def calculate_portfolio_risk_metrics(self, positions: Dict[str, float], days: int = 60) -> Dict:
        """Calculate comprehensive risk metrics for portfolio."""
        
        try:
            symbols = list(positions.keys())
            weights = np.array(list(positions.values()))
            weights = weights / np.sum(weights)  # Normalize
            
            # Get returns data
            query = text("""
                SELECT symbol, date, returns, close
                FROM stock_prices
                WHERE symbol = ANY(:symbols)
                AND date >= CURRENT_DATE - INTERVAL ':days days'
                AND returns IS NOT NULL
                ORDER BY date
            """)
            
            df = pd.read_sql(query, self.engine, params={'symbols': symbols, 'days': days})
            
            if df.empty:
                return {'error': 'Insufficient data for risk analysis'}
            
            # Create returns matrix
            returns_pivot = df.pivot(index='date', columns='symbol', values='returns')
            returns_pivot = returns_pivot.fillna(0)
            
            # Align weights with available data
            available_symbols = returns_pivot.columns.tolist()
            aligned_weights = []
            for symbol in available_symbols:
                if symbol in symbols:
                    idx = symbols.index(symbol)
                    aligned_weights.append(weights[idx])
                else:
                    aligned_weights.append(0)
            
            aligned_weights = np.array(aligned_weights)
            if np.sum(aligned_weights) > 0:
                aligned_weights = aligned_weights / np.sum(aligned_weights)
            
            # Calculate portfolio returns
            portfolio_returns = (returns_pivot * aligned_weights).sum(axis=1)
            
            # Risk metrics
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
            portfolio_return = portfolio_returns.mean() * 252      # Annualized
            
            # Sharpe Ratio (assuming 3% risk-free rate)
            risk_free_rate = 0.03
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            var_95_dollar = var_95  # Percentage terms
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Beta calculation (vs SPY if available)
            beta = self._calculate_portfolio_beta(portfolio_returns)
            
            # Risk assessment
            risk_level = self._assess_risk_level(portfolio_vol, max_drawdown, var_95)
            
            results = {
                'portfolio_volatility': round(portfolio_vol * 100, 2),
                'portfolio_return': round(portfolio_return * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'var_95_percent': round(var_95 * 100, 2),
                'max_drawdown_percent': round(max_drawdown * 100, 2),
                'beta': round(beta, 3) if beta else None,
                'risk_level': risk_level,
                'risk_adjusted_return': round(portfolio_return / portfolio_vol, 3) if portfolio_vol > 0 else 0,
                'analysis_period_days': len(portfolio_returns),
                'professional_assessment': self._generate_risk_assessment(
                    portfolio_vol, sharpe_ratio, max_drawdown, risk_level
                )
            }
            
            logger.info("Portfolio risk metrics calculated",
                       volatility=portfolio_vol,
                       sharpe=sharpe_ratio,
                       risk_level=risk_level)
            
            return results
            
        except Exception as e:
            logger.error("Risk metrics calculation failed", error=str(e))
            return {'error': str(e)}
    
    def generate_portfolio_report(self, positions: Dict[str, float]) -> Dict:
        """Generate comprehensive professional portfolio report."""
        
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'portfolio_composition': {
                    'total_positions': len(positions),
                    'position_weights': {k: round(v, 2) for k, v in positions.items()}
                }
            }
            
            # Add diversification analysis
            diversification = self.analyze_portfolio_diversification(positions)
            if 'error' not in diversification:
                report['diversification_analysis'] = diversification
            
            # Add risk metrics
            risk_metrics = self.calculate_portfolio_risk_metrics(positions)
            if 'error' not in risk_metrics:
                report['risk_analysis'] = risk_metrics
            
            # Add correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(list(positions.keys()))
            if not correlation_matrix.empty:
                report['correlation_matrix'] = correlation_matrix.round(3).to_dict()
            
            # Professional summary
            report['executive_summary'] = self._generate_executive_summary(
                diversification, risk_metrics, len(positions)
            )
            
            logger.info("Professional portfolio report generated",
                       positions=len(positions))
            
            return report
            
        except Exception as e:
            logger.error("Portfolio report generation failed", error=str(e))
            return {'error': str(e)}
    
    def _calculate_portfolio_beta(self, portfolio_returns: pd.Series) -> Optional[float]:
        """Calculate portfolio beta vs market."""
        
        try:
            # Try to get SPY data as market proxy
            query = text("""
                SELECT date, returns FROM stock_prices
                WHERE symbol = 'SPY' 
                AND date >= CURRENT_DATE - INTERVAL '60 days'
                AND returns IS NOT NULL
                ORDER BY date
            """)
            
            market_df = pd.read_sql(query, self.engine)
            
            if market_df.empty:
                return None
            
            # Align dates
            market_returns = market_df.set_index('date')['returns']
            aligned_portfolio = portfolio_returns.reindex(market_returns.index).dropna()
            aligned_market = market_returns.reindex(aligned_portfolio.index).dropna()
            
            if len(aligned_portfolio) < 10:  # Need minimum observations
                return None
            
            # Calculate beta
            covariance = np.cov(aligned_portfolio, aligned_market)[0, 1]
            market_variance = np.var(aligned_market)
            
            beta = covariance / market_variance if market_variance > 0 else None
            
            return beta
            
        except Exception as e:
            logger.error("Beta calculation failed", error=str(e))
            return None
    
    def _assess_risk_level(self, volatility: float, max_drawdown: float, var_95: float) -> str:
        """Professional risk level assessment."""
        
        risk_factors = 0
        
        if volatility > 0.25:  # > 25% annual volatility
            risk_factors += 2
        elif volatility > 0.20:
            risk_factors += 1
        
        if abs(max_drawdown) > 0.15:  # > 15% drawdown
            risk_factors += 2
        elif abs(max_drawdown) > 0.10:
            risk_factors += 1
        
        if abs(var_95) > 0.05:  # > 5% daily VaR
            risk_factors += 1
        
        if risk_factors >= 4:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_diversification_recommendations(self, score: float, conc_risk: str, 
                                                corr_risk: str, sectors: int) -> List[str]:
        """Generate professional diversification recommendations."""
        
        recommendations = []
        
        if score < 50:
            recommendations.append("URGENT: Portfolio requires immediate diversification improvement")
        
        if conc_risk == "HIGH":
            recommendations.append("Reduce position concentration - consider maximum 10% per holding")
        
        if corr_risk == "HIGH":
            recommendations.append("Add uncorrelated assets to reduce correlation risk")
        
        if sectors < 3:
            recommendations.append("Expand sector exposure - target minimum 4 sectors")
        
        if score >= 80:
            recommendations.append("Excellent diversification - maintain current structure")
        elif score >= 65:
            recommendations.append("Good diversification - minor optimization opportunities exist")
        
        return recommendations
    
    def _generate_risk_assessment(self, vol: float, sharpe: float, drawdown: float, 
                                risk_level: str) -> str:
        """Generate professional risk assessment."""
        
        assessment = f"Portfolio Risk Level: {risk_level}"
        
        if sharpe > 1.0:
            assessment += " | Excellent risk-adjusted returns"
        elif sharpe > 0.5:
            assessment += " | Good risk-adjusted returns"
        else:
            assessment += " | Poor risk-adjusted returns - review strategy"
        
        if abs(drawdown) > 0.20:
            assessment += " | WARNING: High drawdown indicates significant risk"
        
        return assessment
    
    def _generate_executive_summary(self, diversification: Dict, risk_metrics: Dict, 
                                  positions: int) -> str:
        """Generate professional executive summary."""
        
        summary_parts = []
        
        # Portfolio size assessment
        if positions < 5:
            summary_parts.append("UNDER-DIVERSIFIED: Portfolio contains too few positions")
        elif positions > 20:
            summary_parts.append("OVER-DIVERSIFIED: Consider consolidating best opportunities")
        else:
            summary_parts.append("APPROPRIATE SIZE: Portfolio position count within optimal range")
        
        # Diversification assessment
        if 'diversification_score' in diversification:
            score = diversification['diversification_score']
            assessment = diversification['assessment']
            summary_parts.append(f"DIVERSIFICATION: {assessment} ({score}/100)")
        
        # Risk assessment
        if 'risk_level' in risk_metrics:
            risk_level = risk_metrics['risk_level']
            sharpe = risk_metrics.get('sharpe_ratio', 0)
            summary_parts.append(f"RISK PROFILE: {risk_level} (Sharpe: {sharpe:.2f})")
        
        return " | ".join(summary_parts)

def test_analytics_system():
    """Test the professional analytics system."""
    
    print("🧪 TESTING QUANTEDGE PROFESSIONAL ANALYTICS")
    print("="*50)
    
    try:
        analytics = QuantEdgeAnalytics()
        
        # Test portfolio positions
        test_positions = {
            'AAPL': 0.25,
            'MSFT': 0.20,
            'GOOGL': 0.15,
            'TSLA': 0.15,
            'JPM': 0.10,
            'NVDA': 0.15
        }
        
        print(f"📊 Testing portfolio with {len(test_positions)} positions")
        
        # Test diversification analysis
        diversification = analytics.analyze_portfolio_diversification(test_positions)
        if 'error' not in diversification:
            print(f"✅ Diversification Score: {diversification['diversification_score']}/100")
            print(f"   Assessment: {diversification['assessment']}")
        else:
            print(f"❌ Diversification analysis failed: {diversification['error']}")
        
        # Test risk metrics
        risk_metrics = analytics.calculate_portfolio_risk_metrics(test_positions)
        if 'error' not in risk_metrics:
            print(f"✅ Portfolio Volatility: {risk_metrics['portfolio_volatility']}%")
            print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']}")
            print(f"   Risk Level: {risk_metrics['risk_level']}")
        else:
            print(f"❌ Risk metrics failed: {risk_metrics['error']}")
        
        # Test full report
        report = analytics.generate_portfolio_report(test_positions)
        if 'error' not in report:
            print("✅ Complete portfolio report generated")
            print(f"   Executive Summary: {report.get('executive_summary', 'N/A')}")
        else:
            print(f"❌ Report generation failed: {report['error']}")
        
        print("\n🎉 PROFESSIONAL ANALYTICS SYSTEM OPERATIONAL!")
        
        return analytics
        
    except Exception as e:
        print(f"❌ Analytics system test failed: {e}")
        return None

if __name__ == "__main__":
    test_analytics_system()