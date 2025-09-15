"""
Renaissance Enhanced Trade Logger
Comprehensive Excel logging with signal breakdown and alternative data tracking.
"""

import os
import pandas as pd
import datetime
import uuid
import json
from typing import Dict, Any, Optional
from pathlib import Path


class RenaissanceTradeLogger:
    """Enhanced trade logger with Renaissance Technologies integration"""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        self.config = {
            'base_dir': 'RenaissanceBotData',
            'excel_filename': 'renaissance_full_log.xlsx',
            'sheet_name': 'RenaissanceLog',
            'dashboard_file': 'dashboard_export.json'
        }

        if custom_config:
            self.config.update(custom_config)

        self._setup_paths()

    def log_renaissance_cycle(self,
                              cycle_data: Dict[str, Any],
                              signal_result: Dict[str, Any],
                              signal_components: Dict[str, float],
                              regime_data: Dict[str, Any],
                              alternative_data: Dict[str, Any],
                              execution_result: Dict[str, Any],
                              position_data: Dict[str, Any]) -> bool:
        """Log complete Renaissance trading cycle"""

        try:
            # Create comprehensive log entry
            log_entry = {
                # Basic cycle information
                'CycleID': cycle_data.get('cycle_id', str(uuid.uuid4())),
                'Timestamp': datetime.datetime.now().isoformat(),
                'RunNumber': cycle_data.get('run_number', 0),

                # Market data
                'Price': cycle_data.get('current_price', 0.0),
                'Volume': cycle_data.get('volume', 0.0),
                'Open': cycle_data.get('open', 0.0),
                'High': cycle_data.get('high', 0.0),
                'Low': cycle_data.get('low', 0.0),
                'Close': cycle_data.get('close', 0.0),

                # Renaissance signal information
                'Signal_Action': signal_result.get('action', 'HOLD'),
                'Signal_Confidence': signal_result.get('confidence', 0.0),
                'Signal_Strength': signal_result.get('strength', 0.0),
                'Signal_Risk_Score': signal_result.get('risk_score', 0.0),

                # Renaissance component breakdown (research-optimized weights)
                'OrderFlow_Score': signal_components.get('order_flow', 0.0),
                'OrderFlow_Weight': '30-34%',
                'OrderBook_Score': signal_components.get('order_book', 0.0),
                'OrderBook_Weight': '18-24%',
                'Volume_Score': signal_components.get('volume', 0.0),
                'Volume_Weight': '10-18%',
                'MACD_Score': signal_components.get('macd', 0.0),
                'MACD_Weight': '8-13%',
                'RSI_Score': signal_components.get('rsi', 0.0),
                'RSI_Weight': '5-18%',
                'Bollinger_Score': signal_components.get('bollinger', 0.0),
                'Bollinger_Weight': '5-18%',
                'Renaissance_Score': signal_components.get('renaissance', 0.0),
                'Renaissance_Weight': '2-7%',

                # Market regime information
                'Market_Regime': regime_data.get('regime', 'unknown'),
                'Sub_Regime': regime_data.get('sub_regime', 'unknown'),
                'Trend_Strength': regime_data.get('trend_strength', 0.0),
                'Volatility_Percentile': regime_data.get('volatility_percentile', 0.0),
                'Volume_Profile': regime_data.get('volume_profile', 'unknown'),

                # Alternative data status
                'Fear_Greed_Index': alternative_data.get('fear_greed', 0.0),
                'Social_Sentiment': alternative_data.get('social_sentiment', 0.0),
                'Market_Momentum': alternative_data.get('market_momentum', 0.0),
                'Alt_Data_Confidence': alternative_data.get('confidence', 0.0),
                'Alt_Data_Status': alternative_data.get('status', 'unknown'),

                # Execution information
                'Trade_Executed': execution_result.get('executed', False),
                'Execution_Reason': execution_result.get('reason', 'No action'),
                'Order_Type': execution_result.get('order_type', 'NONE'),
                'Trade_Size': execution_result.get('size', 0.0),
                'Execution_Price': execution_result.get('price', 0.0),

                # Position information
                'Current_Position': position_data.get('side', 'NONE'),
                'Position_Size': position_data.get('size', 0.0),
                'Unrealized_PnL': position_data.get('unrealized_pnl', 0.0),
                'Realized_PnL': position_data.get('realized_pnl', 0.0),
                'Daily_PnL': position_data.get('daily_pnl', 0.0),

                # Performance metrics
                'Win_Rate': position_data.get('win_rate', 0.0),
                'Total_Trades': position_data.get('total_trades', 0),
                'Winning_Trades': position_data.get('winning_trades', 0),
                'Losing_Trades': position_data.get('losing_trades', 0),

                # System information
                'Execution_Time_Ms': cycle_data.get('execution_time', 0.0) * 1000,
                'Memory_Usage_MB': cycle_data.get('memory_usage', 0.0),
                'CPU_Usage_Percent': cycle_data.get('cpu_usage', 0.0),

                # Decision reasoning
                'Decision_Reasons': '; '.join(signal_result.get('reasons', [])),
                'Risk_Factors': json.dumps(signal_result.get('risk_factors', {})),

                # Metadata
                'Data_Version': 'Renaissance_v1.0',
                'Config_Hash': cycle_data.get('config_hash', 'unknown')
            }

            # Save to Excel
            success = self._save_to_excel(log_entry)

            # Export dashboard data
            if success:
                self._export_dashboard_data(log_entry)

            return success

        except Exception as e:
            print(f"Failed to log Renaissance cycle: {e}")
            return False

    def _save_to_excel(self, log_entry: Dict[str, Any]) -> bool:
        """Save log entry to Excel file"""

        try:
            df = pd.DataFrame([log_entry])

            if self.excel_file_path.exists():
                # Append to existing file
                existing_df = pd.read_excel(self.excel_file_path, sheet_name=self.config['sheet_name'])
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df

            combined_df.to_excel(
                self.excel_file_path,
                sheet_name=self.config['sheet_name'],
                index=False
            )

            return True

        except Exception as e:
            print(f"Failed to save to Excel: {e}")
            return False

    def _export_dashboard_data(self, log_entry: Dict[str, Any]):
        """Export data for dashboard consumption"""

        try:
            dashboard_data = {
                "last_update": datetime.datetime.now().isoformat(),
                "current_signal": {
                    "action": log_entry['Signal_Action'],
                    "confidence": log_entry['Signal_Confidence'],
                    "strength": log_entry['Signal_Strength']
                },
                "signal_breakdown": {
                    "order_flow": {"score": log_entry['OrderFlow_Score'], "weight": log_entry['OrderFlow_Weight']},
                    "order_book": {"score": log_entry['OrderBook_Score'], "weight": log_entry['OrderBook_Weight']},
                    "volume": {"score": log_entry['Volume_Score'], "weight": log_entry['Volume_Weight']},
                    "renaissance": {"score": log_entry['Renaissance_Score'], "weight": log_entry['Renaissance_Weight']}
                },
                "market_regime": {
                    "current": log_entry['Market_Regime'],
                    "sub_regime": log_entry['Sub_Regime'],
                    "trend_strength": log_entry['Trend_Strength']
                },
                "alternative_data": {
                    "fear_greed": log_entry['Fear_Greed_Index'],
                    "social_sentiment": log_entry['Social_Sentiment'],
                    "status": log_entry['Alt_Data_Status']
                },
                "positions": {
                    "current": log_entry['Current_Position'],
                    "size": log_entry['Position_Size'],
                    "pnl": {
                        "unrealized": log_entry['Unrealized_PnL'],
                        "realized": log_entry['Realized_PnL'],
                        "daily": log_entry['Daily_PnL']
                    }
                },
                "performance": {
                    "win_rate": log_entry['Win_Rate'],
                    "total_trades": log_entry['Total_Trades'],
                    "winning_trades": log_entry['Winning_Trades']
                }
            }

            dashboard_file = self.excel_file_path.parent / self.config['dashboard_file']
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Failed to export dashboard data: {e}")


# Global convenience function
def log_renaissance_trading_cycle(**kwargs) -> bool:
    """Convenience function for logging Renaissance trading cycles"""
    logger = RenaissanceTradeLogger()
    return logger.log_renaissance_cycle(**kwargs)
def log_comprehensive_data(data_dict, sub_scores=None, pos_manager=None): 
    # Removed circular import
    logger_instance = get_trade_logger()
    return logger_instance.log_comprehensive_data(data_dict, sub_scores, pos_manager)
def log_comprehensive_data(data_dict, sub_scores=None, pos_manager=None): 
    # Removed circular import
    logger_instance = get_trade_logger()
    return logger_instance.log_comprehensive_data(data_dict, sub_scores, pos_manager)
