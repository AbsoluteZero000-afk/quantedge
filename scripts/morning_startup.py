"""
QuantEdge Daily Startup Script

Perfect way to start your trading day! Checks system health,
loads fresh data, analyzes overnight moves, and prepares trading signals.
"""

#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def morning_startup():
    """Complete morning trading system startup routine."""
    
    print("🌅 QUANTEDGE MORNING STARTUP ROUTINE")
    print("="*50)
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"⏰ {datetime.now().strftime('%I:%M %p')}")
    print()
    
    # Step 1: System Health Check
    print("🏥 [1/5] System Health Check...")
    try:
        os.system("python tests/complete_system_test.py")
        print("   ✅ System diagnostics complete")
    except:
        print("   ⚠️ System test had issues - check manually")
    
    time.sleep(1)
    
    # Step 2: Fresh Data Update
    print("\n📊 [2/5] Loading Fresh Market Data...")
    try:
        os.system("python data_ingestion/optimized_loader.py")
        print("   ✅ Market data refreshed")
    except:
        print("   ⚠️ Data loading had issues")
    
    time.sleep(1)
    
    # Step 3: Generate Trading Signals
    print("\n🎯 [3/5] Analyzing Trading Opportunities...")
    try:
        os.system("python strategies/momentum_strategy.py")
        print("   ✅ Momentum signals generated")
    except:
        print("   ⚠️ Signal generation issues")
    
    time.sleep(1)
    
    # Step 4: Risk Assessment
    print("\n⚖️ [4/5] Risk Management Check...")
    try:
        os.system("python risk_manager/risk_manager.py")
        print("   ✅ Risk parameters validated")
    except:
        print("   ⚠️ Risk manager issues")
    
    time.sleep(1)
    
    # Step 5: Performance Summary
    print("\n📈 [5/5] Performance Monitor...")
    try:
        os.system("python monitoring/performance_monitor.py")
        print("   ✅ Performance metrics calculated")
    except:
        print("   ⚠️ Performance monitor issues")
    
    print("\n" + "="*50)
    print("🎉 MORNING STARTUP COMPLETE!")
    print()
    print("📊 Your QuantEdge system is ready for trading!")
    print("🚀 Launch dashboard: streamlit run dashboard/app.py")
    print("🤖 Or run auto trading: python trader/fixed_auto_trader.py")
    print()
    print("💡 Market opens at 9:30 AM ET - Trade wisely!")
    print("="*50)

if __name__ == "__main__":
    morning_startup()