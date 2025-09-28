"""
QuantEdge Streamlit Dashboard

Interactive web dashboard for portfolio monitoring and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="QuantEdge Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def create_equity_curve_chart():
    """Create sample equity curve chart."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, 252)
    equity_curve = pd.Series(100000 * np.cumprod(1 + returns), index=dates)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        name="Portfolio",
        line=dict(color="#1f77b4", width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400
    )
    
    return fig


def main():
    """Main dashboard application."""
    # Header
    st.title("📈 QuantEdge Dashboard")
    st.markdown("**Personal Mini Hedge Fund Management System**")
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.subheader("📊 System Status")
    
    # System status
    st.sidebar.text("✅ Dashboard")
    st.sidebar.text("❌ Data Ingester (needs FMP key)")
    st.sidebar.text("✅ Risk Manager")
    st.sidebar.text("✅ Backtester")
    
    if st.sidebar.button("Refresh Data", type="primary"):
        st.sidebar.success("✅ Data refreshed")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Backtesting", "⚖️ Risk", "💰 Trading"])
    
    with tab1:
        st.header("📊 Portfolio Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", "$100,000", delta="$0")
        
        with col2:
            st.metric("Buying Power", "$100,000")
        
        with col3:
            st.metric("Positions", "0")
        
        with col4:
            st.metric("Daily P&L", "$0", delta="$0")
        
        # Chart
        fig = create_equity_curve_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Instructions
        st.info("""
        🚀 **Welcome to QuantEdge!** Your personal hedge fund system is ready.
        
        **Next Steps:**
        1. Get a Financial Modeling Prep API key and add it to your .env file
        2. Set up PostgreSQL database
        3. Run data ingestion to populate price data
        4. Start backtesting your strategies
        """)
    
    with tab2:
        st.header("🔬 Strategy Backtesting")
        
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox("Strategy", ["Momentum Rotation", "Factor Screening"])
        with col2:
            timeframe = st.selectbox("Timeframe", ["1 Year", "2 Years", "5 Years"])
        
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                # Mock backtest results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", "12.5%")
                with col2:
                    st.metric("Sharpe Ratio", "1.8")
                with col3:
                    st.metric("Max Drawdown", "-8.2%")
    
    with tab3:
        st.header("⚖️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Parameters")
            max_position = st.slider("Max Position Size", 0.01, 0.20, 0.10)
            kelly_fraction = st.slider("Kelly Fraction", 0.10, 1.00, 0.25)
        
        with col2:
            st.subheader("Current Risk Metrics")
            st.metric("Portfolio Risk", "0%")
            st.metric("Largest Position", "0%")
            st.metric("Risk Budget Used", "0%")
    
    with tab4:
        st.header("💰 Trading Interface")
        
        st.warning("⚠️ Trading interface will be available after API setup")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", "AAPL")
        with col2:
            quantity = st.number_input("Quantity", 1)
        with col3:
            side = st.selectbox("Side", ["BUY", "SELL"])
        
        if st.button("Submit Order (Demo Mode)"):
            st.success(f"Demo order: {side} {quantity} shares of {symbol}")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            QuantEdge v1.0.0 | Built for systematic trading
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
