#!/usr/bin/env python3
"""
Main script to run backtesting on trading signals.
"""

import sys
import os
import glob
import pandas as pd
import yaml
from datetime import datetime

# Import from same src directory
from backtesting import BacktestEngine, PerformanceCalculator
from backtesting.trade import Trade


class DailyTradeLogger:
    """Logs trade entries and exits for the current test_end_date only."""
    
    def __init__(self):
        # Read test_end_date from config
        config_path = "config/parameters.yml"
        if not os.path.exists(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, '..', config_path)
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        data_config = config.get('data', {})
        self.test_end_date = data_config.get('test_end_date', datetime.now().strftime('%Y-%m-%d'))

        # self.test_end_date = config.get('data', datetime.now().strftime('%Y-%m-%d'))
        self.log_file_path = "reports/trades/daily_trades.csv"
        
        # Create the log file with headers if it doesn't exist
        if not os.path.exists(self.log_file_path):
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            headers = ['Date', 'Model', 'Instrument_Token', 'Action', 'Price', 'PnL', 'Reason', 'Signal_Prob']
            pd.DataFrame(columns=headers).to_csv(self.log_file_path, index=False)
    
    def log_trade_entry(self, date: str, model: str, instrument_token: int, entry_price: float, signal_prob: float = None):
        """Log a trade entry only if it occurs on test_end_date."""
        if date == self.test_end_date:
            activity = {
                'Date': date,
                'Model': model,
                'Instrument_Token': instrument_token,
                'Action': 'ENTRY',
                'Price': entry_price,
                'PnL': 0.0,
                'Reason': 'Signal',
                'Signal_Prob': signal_prob
            }
            self._append_to_csv(activity)
    
    def log_trade_exit(self, date: str, model: str, instrument_token: int, exit_price: float, pnl: float, reason: str, signal_prob: float = None):
        """Log a trade exit only if it occurs on test_end_date."""
        if date == self.test_end_date:
            activity = {
                'Date': date,
                'Model': model,
                'Instrument_Token': instrument_token,
                'Action': 'EXIT',
                'Price': exit_price,
                'PnL': pnl,
                'Reason': reason,
                'Signal_Prob': signal_prob
            }
            self._append_to_csv(activity)
    
    def log_signal_no_execution(self, signal_date: str, model: str, instrument_token: int, signal_prob: float = None):
        """Log a signal that couldn't be executed due to missing price data."""
        if signal_date == self.test_end_date:
            activity = {
                'Date': signal_date,
                'Model': model,
                'Instrument_Token': instrument_token,
                'Action': 'SIGNAL_NO_EXECUTION',
                'Price': None,
                'PnL': 0.0,
                'Reason': 'missing_price_data',
                'Signal_Prob': signal_prob
            }
            self._append_to_csv(activity)
    
    def _append_to_csv(self, activity: dict):
        """Append activity to CSV file."""
        df = pd.DataFrame([activity])
        df.to_csv(self.log_file_path, mode='a', header=False, index=False)


def main():
    """Run the complete backtesting process for all signal files."""
    print("="*80)
    print("MULTI-MODEL ALGORITHMIC TRADING BACKTESTING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all signal files
    signals_pattern = "data/signals/*_up_*_thresh_*_signals.csv"
    signal_files = glob.glob(signals_pattern)
    
    if not signal_files:
        print(f"❌ No signal files found matching pattern: {signals_pattern}")
        return 1
    
    print(f"\n📁 Found {len(signal_files)} signal files to backtest:")
    for file in signal_files:
        print(f"  - {os.path.basename(file)}")
    
    # Initialize daily trade logger
    trade_logger = DailyTradeLogger()
    
    # Store all results
    all_results = {}
    
    try:
        # Process each signal file
        for i, signals_file in enumerate(signal_files):
            model_name = os.path.basename(signals_file).replace('_signals.csv', '')
            print(f"\n{'='*60}")
            print(f"BACKTESTING MODEL {i+1}/{len(signal_files)}: {model_name}")
            print(f"{'='*60}")
            
            # Initialize backtesting engine with logging
            engine = BacktestEngineWithLogging(trade_logger, model_name)
            
            # Load signals data
            engine.load_signals_data(signals_file)
            
            # Prepare price data
            engine.prepare_price_data()
            
            # Run backtest
            engine.run_backtest()
            
            # Calculate performance metrics
            if engine.completed_trades:
                performance_calc = PerformanceCalculator(engine.completed_trades)
                metrics = performance_calc.calculate_all_metrics()
                all_results[model_name] = metrics
                
                # Print individual model summary
                performance_calc.print_performance_summary()
            else:
                print(f"\n❌ No trades generated for {model_name}")
                all_results[model_name] = None
        
        # Create consolidated report
        print(f"\n{'='*80}")
        print("CONSOLIDATED BACKTESTING RESULTS")
        print(f"{'='*80}")
        
        if all_results:
            # Create consolidated DataFrame
            consolidated_data = []
            for model_name, metrics in all_results.items():
                if metrics:
                    row = {
                        'Model': model_name,
                        'Total_Trades': metrics['total_trades'],
                        'Win_Rate_%': metrics['win_rate_pct'],
                        'Total_PnL': metrics['total_pnl'],
                        'Profit_Factor': metrics['profit_factor'],
                        'Avg_Trade_PnL': metrics['avg_trade_pnl'],
                        'Avg_Return_%': metrics['avg_return_pct'],
                        'Max_Drawdown_PnL': metrics['max_drawdown_pnl'],
                        'Max_Drawdown_%': metrics['max_drawdown_pct'],
                        'Best_Trade_PnL': metrics['best_trade_pnl'],
                        'Worst_Trade_PnL': metrics['worst_trade_pnl'],
                        'Target_Hit_Count': metrics.get('target_hit_count', 0),
                        'Stop_Loss_Count': metrics.get('stop_loss_count', 0),
                        'Holding_Period_Count': metrics.get('holding_period_count', 0)
                    }
                    consolidated_data.append(row)
            
            if consolidated_data:
                df = pd.DataFrame(consolidated_data)
                
                # Sort by profit factor (best performing first)
                df = df.sort_values('Profit_Factor', ascending=False)
                
                # Print consolidated table
                print(f"\n📊 PERFORMANCE COMPARISON (Ranked by Profit Factor):")
                print("-" * 120)
                
                # Create formatted display
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 200)
                pd.set_option('display.max_colwidth', 25)
                
                print(df.to_string(index=False, float_format='{:.2f}'.format))
                
                # Highlight best performers
                print(f"\n🏆 TOP PERFORMERS:")
                print(f"  Best Profit Factor: {df.iloc[0]['Model']} ({df.iloc[0]['Profit_Factor']:.3f})")
                print(f"  Best Win Rate: {df.loc[df['Win_Rate_%'].idxmax()]['Model']} ({df['Win_Rate_%'].max():.2f}%)")
                print(f"  Best Total PnL: {df.loc[df['Total_PnL'].idxmax()]['Model']} (₹{df['Total_PnL'].max():,.2f})")
                print(f"  Lowest Drawdown: {df.loc[df['Max_Drawdown_%'].idxmax()]['Model']} ({df['Max_Drawdown_%'].max():.2f}%)")
                
                # Export consolidated results
                reports_dir = "reports/backtest_results"
                os.makedirs(reports_dir, exist_ok=True)
                consolidated_file = f"{reports_dir}/consolidated_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(consolidated_file, index=False)
                
                print(f"\n📁 Consolidated results exported to: {consolidated_file}")
            else:
                print("\n❌ No successful backtests to consolidate")
        
        print(f"\n✅ Multi-model backtesting completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


class BacktestEngineWithLogging(BacktestEngine):
    """Extended BacktestEngine that logs trade activities."""
    
    def __init__(self, trade_logger: DailyTradeLogger, model_name: str, config_path: str = "config/parameters.yml"):
        super().__init__(config_path)
        self.trade_logger = trade_logger
        self.model_name = model_name
    
    def run_backtest(self):
        """Run backtest with trade logging."""
        if self.signals_df is None or self.price_df is None:
            raise ValueError("Signals and price data must be loaded before running backtest")
        
        # Get unique instruments and dates
        unique_instruments = sorted(self.signals_df['instrument_token'].unique())
        all_dates = sorted(self.price_df['timestamp'].unique())
        all_dates = all_dates[5:]
        
        total_signals_processed = 0
        trades_entered = 0
        
        # MAIN ITERATION: Instrument by Instrument, Date by Date
        for i, instrument_token in enumerate(unique_instruments):
            active_trade = None
            active_trade_signal_prob = None  # Store signal_prob for the active trade
            instrument_signals = 0
            
            # Get all signals for this instrument
            instrument_signal_data = self.signals_df[
                self.signals_df['instrument_token'] == instrument_token
            ].copy()
            
            for date in all_dates:
                # 1. CHECK EXIT CONDITIONS for active trade
                if active_trade is not None:
                    current_day_data = self.get_current_day_data(instrument_token, date)
                    if current_day_data is not None:
                        should_exit, exit_reason = active_trade.check_exit_condition(
                            date, current_day_data['close']
                        )
                        
                        if should_exit:
                            active_trade.close_trade(date, current_day_data['close'], exit_reason)
                            self.completed_trades.append(active_trade)
                            
                            # Log the exit (use the original signal_prob from entry)
                            self.trade_logger.log_trade_exit(
                                date, self.model_name, instrument_token, 
                                current_day_data['close'], active_trade.pnl, exit_reason,
                                signal_prob=active_trade_signal_prob
                            )
                            
                            active_trade = None
                            active_trade_signal_prob = None
                
                # 2. CHECK FOR NEW SIGNAL ENTRY (only if no active trade)
                elif active_trade is None:
                    # Check if there's a signal for this instrument on this date
                    day_signals = instrument_signal_data[
                        instrument_signal_data['timestamp'] == date
                    ]
                    
                    if not day_signals.empty:
                        signal_data = day_signals.iloc[0]
                        total_signals_processed += 1
                        instrument_signals += 1
                        
                        # Extract signal probability from signal data
                        signal_prob = signal_data.get('signal_prob', None)
                        active_trade_signal_prob = signal_prob  # Store for potential exit logging
                        
                        # Get next day's data for entry
                        next_day_data = self.get_next_day_data(instrument_token, date)
                        
                        if next_day_data is not None:
                            # Enter new trade with actual execution
                            active_trade = Trade(
                                instrument_token=instrument_token,
                                entry_date=next_day_data['timestamp'],  # Actual entry date
                                entry_price=next_day_data['open'],
                                target_price_pct=self.target_price_pct,
                                stop_loss_pct=self.stop_loss_pct,
                                holding_period=self.holding_period
                            )
                            trades_entered += 1
                            
                            # Log the entry with signal probability
                            self.trade_logger.log_trade_entry(
                                next_day_data['timestamp'], self.model_name, 
                                instrument_token, next_day_data['open'], signal_prob
                            )
                        else:
                            # Create trade with no execution (track the signal)
                            active_trade = Trade(
                                instrument_token=instrument_token,
                                entry_date=None,  # No execution possible
                                entry_price=None,
                                target_price_pct=self.target_price_pct,
                                stop_loss_pct=self.stop_loss_pct,
                                holding_period=self.holding_period
                            )
                            
                            # Log the signal that couldn't be executed with signal probability
                            self.trade_logger.log_signal_no_execution(
                                date, self.model_name, instrument_token, signal_prob
                            )
            
            # Close any remaining active trade at the end
            if active_trade is not None:
                if active_trade.execution_status == 'executed':
                    # Find the last available price data for this instrument
                    last_day_data = self.price_df[
                        self.price_df['instrument_token'] == instrument_token
                    ].sort_values('timestamp').iloc[-1]
                    
                    active_trade.close_trade(
                        last_day_data['timestamp'], 
                        last_day_data['close'], 
                        'data_end'
                    )
                    
                    # Log the exit with original signal probability
                    self.trade_logger.log_trade_exit(
                        last_day_data['timestamp'], self.model_name, instrument_token,
                        last_day_data['close'], active_trade.pnl, 'data_end',
                        signal_prob=active_trade_signal_prob
                    )
                else:
                    # No-execution trade - just close it as signal_no_execution
                    active_trade.close_trade(
                        date, 0.0, 'signal_no_execution'
                    )
                
                self.completed_trades.append(active_trade)
        
        print(f"✅ Completed: {total_signals_processed} signals → {len(self.completed_trades)} trades")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 