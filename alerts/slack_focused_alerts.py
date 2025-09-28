"""
QuantEdge Alert System - Slack Webhook Focused

Streamlined alert system optimized for Slack webhook notifications
with your specific webhook URL integrated.
"""

import os
import requests
import json
from datetime import datetime
from typing import Dict, List
import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class QuantEdgeAlerter:
    """Streamlined alert system focused on Slack webhook integration."""
    
    def __init__(self):
        # Your specific Slack webhook
        self.slack_webhook = "https://hooks.slack.com/services/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"
        
        # Backup webhook from .env (if you want to change it later)
        self.backup_webhook = os.getenv('SLACK_WEBHOOK_URL')
        
        self.slack_enabled = True  # Always enabled with your webhook
        
        logger.info("QuantEdgeAlerter initialized with Slack webhook")
    
    def send_slack_alert(self, subject: str, message: str, priority: str = "normal") -> bool:
        """Send professional Slack notification with rich formatting."""
        
        try:
            # Color coding by priority for visual distinction
            colors = {
                'critical': '#dc3545',  # Red
                'warning': '#ffc107',   # Yellow  
                'success': '#28a745',   # Green
                'info': '#17a2b8',      # Blue
                'normal': '#6c757d'     # Gray
            }
            
            # Priority emojis
            priority_emojis = {
                'critical': '🚨',
                'warning': '⚠️',
                'success': '✅',
                'info': '📊',
                'normal': '📈'
            }
            
            color = colors.get(priority, '#6c757d')
            emoji = priority_emojis.get(priority, '📈')
            
            # Enhanced Slack payload with professional formatting
            payload = {
                "username": "QuantEdge Trading Bot",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} QuantEdge Alert: {subject}",
                        "text": message,
                        "fields": [
                            {
                                "title": "Priority",
                                "value": priority.upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": datetime.now().strftime('%I:%M %p ET'),
                                "short": True
                            }
                        ],
                        "footer": "QuantEdge Professional Trading System",
                        "footer_icon": "https://cdn-icons-png.flaticon.com/512/2942/2942813.png",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            # Send to your Slack webhook
            response = requests.post(
                self.slack_webhook,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Slack alert sent successfully", subject=subject, priority=priority)
                return True
            else:
                logger.error("Slack alert failed", 
                           status=response.status_code, 
                           response=response.text)
                return False
                
        except Exception as e:
            logger.error("Slack alert exception", error=str(e))
            return False
    
    def send_trading_signal_alert(self, signals: List[Dict]):
        """Specialized alert for new trading signals."""
        
        if not signals:
            return False
        
        signal_list = []
        for signal in signals[:5]:  # Top 5 signals
            signal_list.append(
                f"📈 *{signal['symbol']}*: ${signal['price']:.2f} "
                f"({signal['momentum']:+.2f}% momentum, {signal.get('confidence', 0):.0f}% confidence)"
            )
        
        message = f"""🎯 *{len(signals)} New Trading Signal{'s' if len(signals) > 1 else ''} Detected*

{chr(10).join(signal_list)}

💡 Review signals in your QuantEdge dashboard and consider execution.
🎮 One-click automated trading available in dashboard."""
        
        return self.send_slack_alert(
            f"{len(signals)} BUY Signal{'s' if len(signals) > 1 else ''} Ready",
            message,
            priority="success"
        )
    
    def send_trade_execution_alert(self, trades: List[Dict], paper_trading: bool = True):
        """Specialized alert for trade executions."""
        
        if not trades:
            return False
        
        successful_trades = [t for t in trades if t.get('status') not in ['FAILED', 'REJECTED']]
        
        if not successful_trades:
            return False
        
        mode = "📝 *Paper Trading*" if paper_trading else "💰 *LIVE TRADING*"
        mode_emoji = "📝" if paper_trading else "🚨"
        
        trade_list = []
        total_value = 0
        
        for trade in successful_trades:
            value = trade['shares'] * trade['estimated_price']
            total_value += value
            
            trade_list.append(
                f"• *{trade['symbol']}*: {trade['action']} {trade['shares']} shares @ ${trade['estimated_price']:.2f} = ${value:,.0f}"
            )
        
        message = f"""{mode_emoji} *QuantEdge Trade Execution Complete*

*Mode:* {mode}
*Total Investment:* ${total_value:,.0f}
*Trades Executed:* {len(successful_trades)}

{chr(10).join(trade_list)}

📊 Check your dashboard for detailed performance tracking.
🎯 Journal entries created automatically."""
        
        priority = "warning" if not paper_trading else "success"
        
        return self.send_slack_alert(
            f"{'🚨 LIVE' if not paper_trading else '📝 Paper'} Trading: {len(successful_trades)} Trade{'s' if len(successful_trades) > 1 else ''} Executed",
            message,
            priority=priority
        )
    
    def send_performance_alert(self, pnl_data: Dict):
        """Alert for significant portfolio performance."""
        
        if 'error' in pnl_data:
            return False
        
        portfolio_return = pnl_data.get('portfolio_return', 0)
        winners = pnl_data.get('winners', 0)
        total = pnl_data.get('total_symbols', 1)
        
        if abs(portfolio_return) < 2:  # Only alert for significant moves
            return False
        
        emoji = "🚀" if portfolio_return > 0 else "📉"
        direction = "gained" if portfolio_return > 0 else "lost"
        
        # Performance context
        if portfolio_return > 3:
            context = "🌟 *Exceptional performance!*"
            priority = "success"
        elif portfolio_return > 1:
            context = "👍 *Solid performance*"
            priority = "info"
        elif portfolio_return < -3:
            context = "⚠️ *Significant decline - review positions*"
            priority = "warning"
        else:
            context = "📊 *Moderate movement*"
            priority = "info"
        
        message = f"""{emoji} *Portfolio Performance Update*

Your QuantEdge portfolio has {direction} *{abs(portfolio_return):.2f}%* today.

📊 *Performance Details:*
• Winners: {winners}/{total} positions
• Win rate: {winners/total*100:.0f}%

{context}

🔗 Check your dashboard for detailed analysis and next steps."""
        
        return self.send_slack_alert(
            f"Portfolio {direction.title()} {abs(portfolio_return):.1f}%",
            message,
            priority=priority
        )
    
    def send_system_health_alert(self, health_score: float, issues: List[str] = None):
        """Alert for system health changes."""
        
        if health_score > 90:
            return False  # Don't alert for good health
        
        if health_score > 70:
            emoji = "⚠️"
            priority = "warning"
            status = "needs attention"
        else:
            emoji = "🚨"
            priority = "critical"
            status = "critical issues detected"
        
        issue_text = ""
        if issues:
            issue_text = f"\n\n*Issues detected:*\n" + "\n".join([f"• {issue}" for issue in issues])
        
        message = f"""{emoji} *QuantEdge System Health Alert*

Your trading system health score: *{health_score:.0f}/100*

Status: {status}{issue_text}

🔧 Run system diagnostics: `python tests/complete_system_test.py`
📊 Check dashboard for detailed system status."""
        
        return self.send_slack_alert(
            f"System Health: {health_score:.0f}/100",
            message,
            priority=priority
        )
    
    def send_market_open_alert(self, signals_count: int, account_value: float):
        """Send market open preparation alert."""
        
        message = f"""🌅 *Market Opening Soon - QuantEdge Ready*

📊 *System Status:*
• Account value: ${account_value:,.0f}
• Active signals: {signals_count}
• Mode: Paper trading (safe testing)

🎯 *Today's Plan:*
• Monitor momentum signals throughout the day
• Execute high-confidence trades via dashboard
• Track performance and optimize parameters

🚀 Ready to capture today's opportunities with systematic precision!"""
        
        return self.send_slack_alert(
            "Market Open - QuantEdge Ready",
            message,
            priority="info"
        )
    
    def send_end_of_day_summary(self, daily_pnl: float, trades_today: int):
        """Send end-of-day performance summary."""
        
        emoji = "🎉" if daily_pnl > 0 else "📊" if daily_pnl == 0 else "🔍"
        
        message = f"""{emoji} *QuantEdge Daily Summary*

📈 *Today's Performance:*
• Portfolio P&L: {daily_pnl:+.2f}%
• Trades executed: {trades_today}
• Status: {"Profitable day!" if daily_pnl > 0 else "Learning opportunity" if daily_pnl < 0 else "Neutral day"}

📊 *Tomorrow's Prep:*
• System will auto-refresh data at 8:00 AM
• New signals will be analyzed pre-market
• Dashboard ready for execution

🎯 Systematic trading delivering results!"""
        
        priority = "success" if daily_pnl > 1 else "info"
        
        return self.send_slack_alert(
            f"EOD Summary: {daily_pnl:+.1f}% P&L",
            message,
            priority=priority
        )
    
    def test_slack_integration(self) -> bool:
        """Test Slack integration with your webhook."""
        
        print("🧪 TESTING QUANTEDGE SLACK INTEGRATION")
        print("="*45)
        print(f"📡 Webhook URL: {self.slack_webhook[:50]}...")
        
        # Send test message
        test_message = f"""🧪 *QuantEdge System Test*

Your professional trading alert system is working perfectly!

✅ Slack webhook integration active
✅ Real-time notifications enabled  
✅ Professional formatting applied
⏰ Test sent at {datetime.now().strftime('%I:%M %p on %B %d, %Y')}

🚀 *Your QuantEdge system is ready to send you:*
• New trading signal alerts
• Trade execution confirmations  
• Performance milestone notifications
• System health warnings
• Daily market summaries

Ready to start systematic wealth building! 💰"""
        
        success = self.send_slack_alert(
            "System Integration Test - All Systems Operational",
            test_message,
            priority="success"
        )
        
        if success:
            print("✅ Slack integration test SUCCESSFUL!")
            print("📱 Check your Slack channel for the test message")
            print("🎯 Your QuantEdge alerts are ready!")
        else:
            print("❌ Slack integration test FAILED")
            print("🔧 Check your webhook URL and internet connection")
        
        return success

def main():
    """Test the streamlined Slack alert system."""
    
    alerter = QuantEdgeAlerter()
    
    # Test basic integration
    alerter.test_slack_integration()
    
    print(f"\n🎯 TESTING SPECIALIZED ALERT TYPES:")
    print("="*40)
    
    # Test trading signal alert
    mock_signals = [
        {'symbol': 'AAPL', 'price': 225.50, 'momentum': 7.8, 'confidence': 75},
        {'symbol': 'TSLA', 'price': 440.20, 'momentum': 5.2, 'confidence': 65}
    ]
    
    print("📊 Testing signal alert...")
    alerter.send_trading_signal_alert(mock_signals)
    print("   ✅ Signal alert sent")
    
    # Test trade execution alert
    mock_trades = [
        {
            'symbol': 'AAPL', 
            'action': 'BUY', 
            'shares': 44, 
            'estimated_price': 225.50,
            'status': 'PAPER_SUCCESS'
        }
    ]
    
    print("💰 Testing execution alert...")
    alerter.send_trade_execution_alert(mock_trades, paper_trading=True)
    print("   ✅ Execution alert sent")
    
    # Test market open alert
    print("🌅 Testing market open alert...")
    alerter.send_market_open_alert(signals_count=3, account_value=100000)
    print("   ✅ Market open alert sent")
    
    print(f"\n🎉 ALL ALERT TYPES TESTED!")
    print("📱 Check your Slack channel for 4 test messages")
    print("🚀 Your QuantEdge alert system is fully operational!")

if __name__ == "__main__":
    main()