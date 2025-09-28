"""
QUANTEDGE MOMENTUM STRATEGY - Your First Trading Algorithm

Analyzes your real stock data and generates momentum-based trading signals
using risk-adjusted scoring and proper position sizing.
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def main():
    """Run momentum analysis on your live stock data."""
    print("🚀 QUANTEDGE MOMENTUM STRATEGY")
    print("="*50)
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not found in .env")
        return
    
    try:
        engine = create_engine(db_url)
        
        # Get your real stock data
        query = """
        SELECT symbol, date, close, volume, returns
        FROM stock_prices 
        WHERE date >= CURRENT_DATE - INTERVAL '20 days'
        ORDER BY symbol, date
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("❌ No data found - run: python data_ingestion/data_ingestion.py")
            return
        
        print(f"📊 Analyzing {len(df)} data points from YOUR database...")
        
        # Calculate momentum for each symbol
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) >= 10:
                prices = symbol_data['close'].values
                
                # Calculate momentum metrics
                mom_5d = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0
                mom_10d = (prices[-1] - prices[-10]) / prices[-10] * 100 if len(prices) >= 10 else 0
                
                # Risk measure (volatility)
                returns = symbol_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Risk-adjusted momentum
                risk_adj_score = mom_10d / volatility if volatility > 0 else 0
                
                results.append({
                    'symbol': symbol,
                    'price': prices[-1],
                    'mom_5d': mom_5d,
                    'mom_10d': mom_10d,
                    'volatility': volatility,
                    'risk_score': risk_adj_score,
                    'data_points': len(symbol_data)
                })
        
        if not results:
            print("❌ Insufficient data for analysis")
            return
        
        # Convert to DataFrame and rank
        df_results = pd.DataFrame(results).sort_values('risk_score', ascending=False)
        
        print(f"\n📈 MOMENTUM RANKINGS:")
        print("-"*60)
        
        for i, (_, row) in enumerate(df_results.iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            trend = "🚀" if row['mom_10d'] > 2 else "📈" if row['mom_10d'] > 0 else "📉"
            
            print(f"{emoji} {row['symbol']}: ${row['price']:.2f}")
            print(f"   {trend} 10d: {row['mom_10d']:+.2f}% | 5d: {row['mom_5d']:+.2f}%")
            print(f"   📊 Risk Score: {row['risk_score']:.3f}")
            print()
        
        # Generate trading signals
        print("🎯 TRADING SIGNALS:")
        print("-"*30)
        
        # Buy top performers with positive momentum
        buy_signals = df_results.head(3)  # Top 3
        buy_signals = buy_signals[buy_signals['mom_10d'] > 0]
        
        if not buy_signals.empty:
            print("🟢 BUY RECOMMENDATIONS:")
            for _, row in buy_signals.iterrows():
                weight = 1.0 / len(buy_signals)
                print(f"   {row['symbol']}: {weight:.1%} allocation")
                print(f"      Momentum: {row['mom_10d']:+.2f}%")
                print(f"      Risk Score: {row['risk_score']:.3f}")
        else:
            print("🔴 NO BUY SIGNALS - Negative momentum across all stocks")
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Analyzed {len(df_results)} stocks")
        print(f"🎯 Generated {len(buy_signals)} buy signals")
        
        return df_results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
