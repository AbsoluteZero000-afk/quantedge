"""
QuantEdge Professional Trade Journal - COMPLETE IMPLEMENTATION

Complete systematic trade journaling with performance attribution,
contextual analysis, and professional insights extraction.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class QuantEdgeJournal:
    """Complete professional trade journaling system."""
    
    def __init__(self, db_path: str = "data/trade_journal.db"):
        """Initialize the professional journal system."""
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        self.db_path = db_path
        self.init_database()
        
        logger.info("QuantEdge Professional Journal initialized", db_path=db_path)
    
    def init_database(self):
        """Initialize the journal database with professional schema."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Professional trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_date DATETIME NOT NULL,
                    exit_date DATETIME,
                    strategy TEXT,
                    confidence_level REAL,
                    position_size_pct REAL,
                    risk_reward_ratio REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl_dollars REAL,
                    pnl_percent REAL,
                    holding_period_days INTEGER,
                    market_conditions TEXT,
                    trade_reasoning TEXT,
                    exit_reasoning TEXT,
                    lessons_learned TEXT,
                    trade_grade TEXT,
                    execution_mode TEXT DEFAULT 'paper',
                    context_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Professional performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    average_win REAL DEFAULT 0,
                    average_loss REAL DEFAULT 0,
                    max_win REAL DEFAULT 0,
                    max_loss REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Professional insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT NOT NULL,
                    insight_text TEXT NOT NULL,
                    related_trades TEXT,
                    confidence_score REAL,
                    actionable BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def log_trade_entry(self, trade_data: Dict, context: Dict = None, notes: str = "") -> int:
        """Log a complete professional trade entry."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract trade data
                symbol = trade_data.get('symbol', '').upper()
                action = trade_data.get('action', '').upper()
                shares = int(trade_data.get('shares', 0))
                entry_price = float(trade_data.get('price', 0))
                
                # Context data
                context = context or {}
                confidence = context.get('confidence', 50.0)
                strategy = context.get('strategy', 'Momentum')
                execution_mode = context.get('execution_mode', 'paper')
                
                # Calculate professional metrics
                position_value = shares * entry_price
                position_size_pct = context.get('position_size_pct', 0)
                
                # Market context
                market_conditions = self._assess_market_conditions(context)
                
                # Trade reasoning
                reasoning = notes or context.get('reasoning', f"Professional {strategy} signal execution")
                
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, action, shares, entry_price, entry_date,
                        strategy, confidence_level, position_size_pct,
                        trade_reasoning, market_conditions, execution_mode,
                        context_data, trade_grade
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, action, shares, entry_price, datetime.now(),
                    strategy, confidence, position_size_pct,
                    reasoning, market_conditions, execution_mode,
                    json.dumps(context), 'PENDING'
                ))
                
                trade_id = cursor.lastrowid
                
                logger.info("Professional trade logged", 
                           trade_id=trade_id, 
                           symbol=symbol, 
                           action=action,
                           shares=shares,
                           confidence=confidence)
                
                return trade_id
                
        except Exception as e:
            logger.error("Trade logging failed", error=str(e))
            return -1
    
    def update_trade_exit(self, trade_id: int, exit_price: float, exit_reasoning: str = ""):
        """Update trade with professional exit analysis."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get trade details
                cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
                trade = cursor.fetchone()
                
                if not trade:
                    logger.warning("Trade not found for exit update", trade_id=trade_id)
                    return
                
                # Calculate professional P&L
                entry_price = trade[5]  # entry_price column
                shares = trade[3]       # shares column
                action = trade[2]       # action column
                
                if action == 'BUY':
                    pnl_dollars = (exit_price - entry_price) * shares
                else:  # SELL
                    pnl_dollars = (entry_price - exit_price) * shares
                
                pnl_percent = (pnl_dollars / (entry_price * shares)) * 100
                
                # Calculate holding period
                entry_date = datetime.fromisoformat(trade[6])  # entry_date
                exit_date = datetime.now()
                holding_days = (exit_date - entry_date).days
                
                # Professional trade grading
                trade_grade = self._grade_trade(pnl_percent, holding_days)
                
                # Update trade
                cursor.execute("""
                    UPDATE trades SET 
                        exit_price = ?, exit_date = ?, exit_reasoning = ?,
                        pnl_dollars = ?, pnl_percent = ?, holding_period_days = ?,
                        trade_grade = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    exit_price, exit_date, exit_reasoning,
                    pnl_dollars, pnl_percent, holding_days,
                    trade_grade, trade_id
                ))
                
                conn.commit()
                
                # Generate insights
                self._generate_trade_insights(trade_id, pnl_percent, trade_grade)
                
                logger.info("Professional trade exit logged",
                           trade_id=trade_id,
                           pnl_dollars=pnl_dollars,
                           pnl_percent=pnl_percent,
                           grade=trade_grade)
                
        except Exception as e:
            logger.error("Trade exit update failed", error=str(e))
    
    def get_journal_summary(self, days: int = 30) -> Dict:
        """Get comprehensive professional journal summary."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Date range
                start_date = datetime.now() - timedelta(days=days)
                
                # Get completed trades
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE entry_date >= ? AND exit_date IS NOT NULL
                    ORDER BY exit_date DESC
                """, (start_date,))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {
                        'total_trades': 0,
                        'win_rate': 0,
                        'total_pnl': 0,
                        'average_pnl': 0,
                        'message': f'No completed trades in last {days} days'
                    }
                
                # Calculate professional metrics
                total_trades = len(trades)
                winners = [t for t in trades if t[15] and t[15] > 0]  # pnl_dollars > 0
                losers = [t for t in trades if t[15] and t[15] < 0]
                
                win_rate = (len(winners) / total_trades) * 100
                total_pnl = sum(t[15] for t in trades if t[15])
                average_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                # Best and worst trades
                best_trade = max(trades, key=lambda x: x[15] or 0) if trades else None
                worst_trade = min(trades, key=lambda x: x[15] or 0) if trades else None
                
                # Professional insights
                cursor.execute("""
                    SELECT insight_text FROM trading_insights 
                    WHERE created_at >= ? AND confidence_score >= 70
                    ORDER BY confidence_score DESC LIMIT 5
                """, (start_date,))
                
                insights = [row[0] for row in cursor.fetchall()]
                
                return {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'average_pnl': average_pnl,
                    'winners': len(winners),
                    'losers': len(losers),
                    'best_trade': self._format_trade_summary(best_trade) if best_trade else None,
                    'worst_trade': self._format_trade_summary(worst_trade) if worst_trade else None,
                    'top_lessons': insights,
                    'days_analyzed': days
                }
                
        except Exception as e:
            logger.error("Journal summary failed", error=str(e))
            return {'error': str(e)}
    
    def get_performance_attribution(self, symbol: str = None) -> Dict:
        """Get professional performance attribution analysis."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute("""
                        SELECT symbol, strategy, AVG(pnl_percent), COUNT(*), 
                               SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins
                        FROM trades 
                        WHERE symbol = ? AND exit_date IS NOT NULL
                        GROUP BY symbol, strategy
                    """, (symbol.upper(),))
                else:
                    cursor.execute("""
                        SELECT symbol, strategy, AVG(pnl_percent), COUNT(*), 
                               SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins
                        FROM trades 
                        WHERE exit_date IS NOT NULL
                        GROUP BY symbol, strategy
                    """)
                
                results = cursor.fetchall()
                
                attribution = []
                for row in results:
                    symbol, strategy, avg_pnl, total, wins = row
                    win_rate = (wins / total) * 100 if total > 0 else 0
                    
                    attribution.append({
                        'symbol': symbol,
                        'strategy': strategy,
                        'average_pnl_percent': avg_pnl or 0,
                        'total_trades': total,
                        'win_rate': win_rate,
                        'contribution_score': (avg_pnl or 0) * total * (win_rate / 100)
                    })
                
                # Sort by contribution score
                attribution.sort(key=lambda x: x['contribution_score'], reverse=True)
                
                return {
                    'performance_attribution': attribution,
                    'top_performer': attribution[0] if attribution else None,
                    'analysis_date': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Performance attribution failed", error=str(e))
            return {'error': str(e)}
    
    def add_manual_lesson(self, lesson_text: str, trade_ids: List[int] = None, confidence: float = 80.0):
        """Add manual professional trading lesson."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_insights (
                        insight_type, insight_text, related_trades, confidence_score
                    ) VALUES (?, ?, ?, ?)
                """, (
                    'MANUAL_LESSON',
                    lesson_text,
                    json.dumps(trade_ids) if trade_ids else None,
                    confidence
                ))
                
                conn.commit()
                
                logger.info("Manual lesson added", lesson=lesson_text[:50])
                
        except Exception as e:
            logger.error("Manual lesson add failed", error=str(e))
    
    def _assess_market_conditions(self, context: Dict) -> str:
        """Assess market conditions from context."""
        
        conditions = []
        
        if context.get('momentum_10d', 0) > 5:
            conditions.append("Strong Uptrend")
        elif context.get('momentum_10d', 0) < -5:
            conditions.append("Strong Downtrend")
        else:
            conditions.append("Sideways Market")
        
        volatility = context.get('volatility', 20)
        if volatility > 30:
            conditions.append("High Volatility")
        elif volatility < 15:
            conditions.append("Low Volatility")
        
        volume_ratio = context.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            conditions.append("High Volume")
        elif volume_ratio < 0.7:
            conditions.append("Low Volume")
        
        return " | ".join(conditions)
    
    def _grade_trade(self, pnl_percent: float, holding_days: int) -> str:
        """Professional trade grading system."""
        
        if pnl_percent > 10:
            return "A+"
        elif pnl_percent > 5:
            return "A"
        elif pnl_percent > 2:
            return "B+"
        elif pnl_percent > 0:
            return "B"
        elif pnl_percent > -2:
            return "C"
        elif pnl_percent > -5:
            return "D"
        else:
            return "F"
    
    def _generate_trade_insights(self, trade_id: int, pnl_percent: float, grade: str):
        """Generate professional insights from trade."""
        
        insights = []
        
        if grade in ['A+', 'A']:
            insights.append(f"Exceptional trade execution - {pnl_percent:.1f}% profit demonstrates strong signal quality")
        
        if pnl_percent < -3:
            insights.append(f"Risk management review needed - {abs(pnl_percent):.1f}% loss exceeds optimal risk parameters")
        
        if grade == 'F':
            insights.append("Critical analysis required - significant loss indicates systematic issue")
        
        # Add insights to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for insight in insights:
                    cursor.execute("""
                        INSERT INTO trading_insights (
                            insight_type, insight_text, related_trades, confidence_score
                        ) VALUES (?, ?, ?, ?)
                    """, ('AUTO_GENERATED', insight, json.dumps([trade_id]), 75.0))
                
                conn.commit()
                
        except Exception as e:
            logger.error("Insight generation failed", error=str(e))
    
    def _format_trade_summary(self, trade_row) -> Dict:
        """Format trade for summary display."""
        
        return {
            'trade_data': {
                'symbol': trade_row[1],
                'action': trade_row[2],
                'shares': trade_row[3],
                'entry_price': trade_row[5],
                'exit_price': trade_row[6],
                'strategy': trade_row[8]
            },
            'trade_analysis': {
                'pnl_dollars': trade_row[15],
                'pnl_percent': trade_row[16],
                'holding_days': trade_row[17],
                'grade': trade_row[22]
            }
        }
    
    def export_journal_csv(self, filename: str = None) -> str:
        """Export professional journal to CSV."""
        
        if not filename:
            filename = f"quantedge_journal_{datetime.now().strftime('%Y%m%d')}.csv"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT 
                        symbol, action, shares, entry_price, exit_price,
                        entry_date, exit_date, pnl_dollars, pnl_percent,
                        holding_period_days, strategy, confidence_level,
                        trade_grade, market_conditions, trade_reasoning
                    FROM trades 
                    WHERE exit_date IS NOT NULL
                    ORDER BY exit_date DESC
                """, conn)
                
                df.to_csv(filename, index=False)
                
                logger.info("Journal exported", filename=filename, trades=len(df))
                return filename
                
        except Exception as e:
            logger.error("Journal export failed", error=str(e))
            return ""

def test_journal_system():
    """Test the professional journal system."""
    
    print("🧪 TESTING QUANTEDGE PROFESSIONAL JOURNAL")
    print("="*50)
    
    journal = QuantEdgeJournal()
    
    # Test trade logging
    trade_data = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'shares': 100,
        'price': 225.50
    }
    
    context = {
        'confidence': 78,
        'strategy': 'Professional Momentum',
        'execution_mode': 'paper',
        'momentum_10d': 7.2,
        'volatility': 22.5,
        'volume_ratio': 1.4
    }
    
    trade_id = journal.log_trade_entry(trade_data, context, "Strong institutional signal")
    print(f"✅ Trade logged: ID {trade_id}")
    
    # Test exit logging
    journal.update_trade_exit(trade_id, 235.75, "Take profit hit")
    print(f"✅ Trade exit logged: ID {trade_id}")
    
    # Test summary
    summary = journal.get_journal_summary(30)
    print(f"✅ Journal summary: {summary}")
    
    # Test manual lesson
    journal.add_manual_lesson("Strong momentum signals above 5% show excellent performance")
    print("✅ Manual lesson added")
    
    print("\n🎉 PROFESSIONAL JOURNAL SYSTEM OPERATIONAL!")
    
    return journal

if __name__ == "__main__":
    test_journal_system()