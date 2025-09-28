# QuantEdge: Personal Mini Hedge Fund System

> **Professional-grade algorithmic trading system for personal wealth acceleration through disciplined quantitative strategies.**

## 🎯 System Overview

QuantEdge combines **long-term investing** with **active quantitative strategies** in a production-ready Python framework. Built with safety, reproducibility, and performance as core principles.

### Key Features

- **🛡️ Risk-First Architecture**: Conservative position sizing with Kelly Criterion
- **📊 Multi-Strategy Framework**: Momentum, factor screening, covered calls  
- **🔄 Real-Time Trading**: Alpaca Markets integration
- **📈 Interactive Dashboard**: Streamlit-based monitoring
- **🗄️ Production Database**: PostgreSQL with comprehensive logging
- **🐳 Containerized**: Docker deployment ready

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL (or Docker)
- Financial Modeling Prep API key
- Alpaca Markets API credentials (included)

### Installation

```bash
# Clone and setup
git clone <your-repo-url>
cd quantedge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Database Setup

```bash
# Option 1: Docker (Recommended)
docker-compose up -d postgres

# Option 2: Local PostgreSQL
createdb quantedge
psql quantedge < database/schema.sql
```

### Running the System

```bash
# Start dashboard
streamlit run dashboard/app.py

# Run data ingestion (separate terminal)
python data_ingestion/data_ingestion.py

# Test backtesting
python backtester/backtester.py
```

## 📊 Strategy Overview

### 1. Momentum Rotation
- Rotate capital into top-performing assets
- Weekly/Monthly rebalancing
- Risk-adjusted position sizing

### 2. Factor Screening  
- Screen stocks on fundamental factors
- Long positions in top-quartile stocks
- Monthly rebalancing

### 3. Covered Calls (Future)
- Generate income on long positions
- Conservative delta targeting
- Automated roll management

## 🛡️ Risk Management

- **Kelly Criterion**: Optimal growth-based sizing (with safety factors)
- **Portfolio Controls**: Maximum 20% total risk, 10% per position
- **Real-time Monitoring**: Daily VaR, drawdown tracking

## 📈 12-Week Development Plan

| Weeks | Focus | Deliverables |
|-------|-------|-------------|
| 1-2 | **Foundation** | Database, FMP integration, data pipeline |
| 3-5 | **Backtesting** | Strategy engine, performance metrics |
| 6-8 | **Trading** | Alpaca integration, paper trading |
| 9-10 | **Dashboard** | Monitoring interface, alerts |
| 11-12 | **Production** | Live trading preparation |

## ⚙️ Configuration

Edit your `.env` file:

```env
# Your Alpaca credentials are already included
FMP_API_KEY=your_fmp_api_key_here
DATABASE_URL=postgresql://user:pass@localhost/quantedge
```

## 🧪 Testing

```bash
# Run tests
pytest

# Check code quality
black .
flake8 .
```

## 📞 Support

- **Issues**: Create GitHub issues for bugs
- **Features**: Submit feature requests
- **Documentation**: Check the wiki

## ⚠️ Disclaimer

**This software is for educational purposes. Trading involves substantial risk of loss. Only trade with capital you can afford to lose. Past performance does not guarantee future results.**

---

**Built for systematic trading success! 🚀**
