"""
QuantEdge Professional Scheduler

Automated scheduling for data refreshes, signal generation,
and system health monitoring. Set it and forget it!
"""

import schedule
import time
import os
import sys
from datetime import datetime
import subprocess
import json
import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class QuantEdgeScheduler:
    """Professional automated scheduling system."""
    
    def __init__(self):
        self.log_file = 'logs/scheduler.json'
        os.makedirs('../logs', exist_ok=True)
        
        # Import alert system
        try:
            sys.path.append('alerts')
            from alert_system import QuantEdgeAlerter
            self.alerter = QuantEdgeAlerter()
            self.alerts_enabled = True
        except:
            self.alerts_enabled = False
    
    def run_data_refresh(self):
        """Scheduled data refresh."""
        
        logger.info("Starting scheduled data refresh")
        
        try:
            # Run optimized data loader
            result = subprocess.run(['python', 'data_ingestion/optimized_loader.py'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Data refresh successful")
                
                if self.alerts_enabled:
                    self.alerter.send_multi_channel_alert(
                        "Daily Data Refresh Complete",
                        "Your QuantEdge system successfully updated market data.",
                        priority="info",
                        channels=['email']
                    )
            else:
                logger.error("Data refresh failed", error=result.stderr)
                
                if self.alerts_enabled:
                    self.alerter.alert_system_error(result.stderr, "Data Refresh")
        
        except Exception as e:
            logger.error("Data refresh exception", error=str(e))
            
            if self.alerts_enabled:
                self.alerter.alert_system_error(str(e), "Scheduled Data Refresh")
        
        self._log_scheduled_task("data_refresh", "completed")
    
    def run_signal_analysis(self):
        """Scheduled signal generation and analysis."""
        
        logger.info("Starting scheduled signal analysis")
        
        try:
            # Run momentum strategy analysis
            result = subprocess.run(['python', 'strategies/momentum_strategy.py'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("Signal analysis successful")
                
                # Check for new high-confidence signals
                # This would integrate with your signal generation logic
                
            else:
                logger.error("Signal analysis failed", error=result.stderr)
        
        except Exception as e:
            logger.error("Signal analysis exception", error=str(e))
        
        self._log_scheduled_task("signal_analysis", "completed")
    
    def run_system_health_check(self):
        """Scheduled comprehensive system health check."""
        
        logger.info("Starting scheduled health check")
        
        try:
            result = subprocess.run(['python', 'tests/complete_system_test.py'], 
                                  capture_output=True, text=True, timeout=180)
            
            # Parse health check results
            health_issues = []
            
            if "FAIL" in result.stdout:
                health_issues.append("System test failures detected")
            
            if "ERROR" in result.stdout:
                health_issues.append("System errors found")
            
            if health_issues and self.alerts_enabled:
                self.alerter.send_multi_channel_alert(
                    "System Health Alert",
                    f"Health check found issues:\n\n" + "\n".join(health_issues),
                    priority="warning",
                    channels=['email', 'slack']
                )
            
            logger.info("Health check completed", issues=len(health_issues))
        
        except Exception as e:
            logger.error("Health check failed", error=str(e))
        
        self._log_scheduled_task("health_check", "completed")
    
    def run_automated_trading(self):
        """Scheduled automated trading execution."""
        
        logger.info("Starting scheduled automated trading")
        
        # Only run during market hours (9:30 AM - 4:00 PM ET)
        current_hour = datetime.now().hour
        
        if not (9 <= current_hour <= 16):
            logger.info("Outside trading hours - skipping automated trading")
            return
        
        # Only run on weekdays
        if datetime.now().weekday() >= 5:
            logger.info("Weekend - skipping automated trading")
            return
        
        try:
            result = subprocess.run(['python', 'trader/fixed_auto_trader.py'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Automated trading completed successfully")
                
                # Parse results for alerts
                if "trades executed" in result.stdout.lower() and self.alerts_enabled:
                    self.alerter.send_multi_channel_alert(
                        "Automated Trading Executed",
                        "Your QuantEdge system executed scheduled trades.\n\nCheck dashboard for details.",
                        priority="success",
                        channels=['email', 'slack']
                    )
            
            else:
                logger.error("Automated trading failed", error=result.stderr)
                
                if self.alerts_enabled:
                    self.alerter.alert_system_error(result.stderr, "Scheduled Trading")
        
        except Exception as e:
            logger.error("Automated trading exception", error=str(e))
        
        self._log_scheduled_task("automated_trading", "completed")
    
    def _log_scheduled_task(self, task_name: str, status: str):
        """Log scheduled task execution."""
        
        try:
            # Load existing log
            try:
                with open(self.log_file, 'r') as f:
                    log = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                log = []
            
            # Add new entry
            log.append({
                'timestamp': datetime.now().isoformat(),
                'task': task_name,
                'status': status
            })
            
            # Keep only last 100 entries
            log = log[-100:]
            
            # Save updated log
            with open(self.log_file, 'w') as f:
                json.dump(log, f, indent=2, default=str)
        
        except Exception as e:
            logger.error("Failed to log scheduled task", error=str(e))
    
    def setup_professional_schedule(self):
        """Setup professional trading schedule."""
        
        print("📅 SETTING UP QUANTEDGE PROFESSIONAL SCHEDULE")
        print("="*55)
        
        # Pre-market data refresh (8:00 AM ET)
        schedule.every().day.at("08:00").do(self.run_data_refresh)
        print("✅ 8:00 AM - Pre-market data refresh")
        
        # Market open signal analysis (9:25 AM ET)  
        schedule.every().day.at("09:25").do(self.run_signal_analysis)
        print("✅ 9:25 AM - Pre-open signal analysis")
        
        # Mid-day automated trading check (12:00 PM ET)
        schedule.every().day.at("12:00").do(self.run_automated_trading)
        print("✅ 12:00 PM - Mid-day trading execution")
        
        # Afternoon signal update (2:00 PM ET)
        schedule.every().day.at("14:00").do(self.run_signal_analysis)
        print("✅ 2:00 PM - Afternoon signal update")
        
        # End of day system health (5:00 PM ET)
        schedule.every().day.at("17:00").do(self.run_system_health_check)
        print("✅ 5:00 PM - End-of-day health check")
        
        # Weekly comprehensive review (Sunday 6:00 PM)
        schedule.every().sunday.at("18:00").do(self.run_weekly_review)
        print("✅ Sunday 6:00 PM - Weekly comprehensive review")
        
        print(f"\n🎯 Professional schedule configured!")
        print(f"📊 6 automated tasks scheduled")
        print(f"🔔 Alerts: {'Enabled' if self.alerts_enabled else 'Disabled'}")
    
    def run_weekly_review(self):
        """Comprehensive weekly performance review."""
        
        logger.info("Starting weekly performance review")
        
        try:
            # Run comprehensive system analysis
            subprocess.run(['python', 'monitoring/performance_monitor.py'], timeout=120)
            
            if self.alerts_enabled:
                self.alerter.send_multi_channel_alert(
                    "Weekly Performance Review",
                    "Your weekly QuantEdge performance review is complete.\n\nCheck dashboard for detailed metrics and insights.",
                    priority="info",
                    channels=['email']
                )
        
        except Exception as e:
            logger.error("Weekly review failed", error=str(e))
    
    def run_scheduler(self):
        """Run the professional scheduler."""
        
        print("🚀 QUANTEDGE PROFESSIONAL SCHEDULER STARTED")
        print("="*50)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Mode: Professional Automated Trading")
        print(f"🔔 Alerts: {'Enabled' if self.alerts_enabled else 'Disabled'}")
        print()
        
        self.setup_professional_schedule()
        
        print(f"\n🏃‍♂️ Scheduler running... Press Ctrl+C to stop")
        print(f"📊 Next task: {schedule.next_run()}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Scheduler stopped by user")
            logger.info("Scheduler stopped by user")

def main():
    """Run the professional scheduler."""
    
    scheduler = QuantEdgeScheduler()
    scheduler.run_scheduler()

if __name__ == "__main__":
    main()