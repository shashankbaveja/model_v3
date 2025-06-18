"""
Main backtesting engine implementing simple iteration approach.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml
import os

from .trade import Trade


class BacktestEngine:
    """Main backtesting engine that processes signals and generates trades."""
    
    def __init__(self, config_path: str = "config/parameters.yml"):
        """
        Initialize the backtesting engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration - adjust path since we're now in src/
        if not os.path.exists(config_path):
            # Try relative path from src directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, '..', '..', config_path)
        
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Extract backtest parameters - use the actual 'backtest' section
        if 'backtest' not in self.config:
            raise KeyError(f"'backtest' section not found in configuration file. Available keys: {list(self.config.keys())}")
        
        backtest_config = self.config['backtest']
        
        # Map the parameter names from config to our expected names
        self.holding_period = backtest_config.get('holding_period', 5)  # Default to 5 if not found
        self.target_price_pct = backtest_config.get('target_price_pct', 3.0)  # Default to 3.0 if not found
        self.stop_loss_pct = backtest_config.get('stop_loss_pct', 1.0)  # Default to 1.0 if not found
        
        # Data storage
        self.signals_df: Optional[pd.DataFrame] = None
        self.price_df: Optional[pd.DataFrame] = None
        self.completed_trades: List[Trade] = []
        self.current_signals_file: Optional[str] = None  # Track current file
        
        print(f"BacktestEngine initialized with:")
        print(f"  Holding Period: {self.holding_period} days")
        print(f"  Target Price: {self.target_price_pct}%")
        print(f"  Stop Loss: {self.stop_loss_pct}%")
    
    def load_signals_data(self, signals_file_path: str):
        """
        Load signals data from CSV file.
        
        Args:
            signals_file_path: Path to signals CSV file
        """
        self.current_signals_file = signals_file_path  # Store current file path
        self.signals_df = pd.read_csv(signals_file_path)
        
        # Convert timestamp to datetime and then to string for consistency
        self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp']).dt.strftime('%Y-%m-%d')
        
        # Filter only buy signals
        self.signals_df = self.signals_df[self.signals_df['signal'] == 1].copy()
        
        print(f"Loaded {len(self.signals_df)} buy signals for {self.signals_df['instrument_token'].nunique()} instruments")
    
    def prepare_price_data(self):
        """
        Prepare price data from signals data.
        For now, we'll use the OHLC data available in the signals file.
        Later this can be enhanced to load from a separate price data source.
        """
        # Create price dataframe with all OHLC data from signals
        # IMPORTANT: Use the ORIGINAL signals_df before filtering, not the filtered one
        # This ensures we have price data for all dates, not just signal dates
        original_signals_df = pd.read_csv(self.current_signals_file)
        original_signals_df['timestamp'] = pd.to_datetime(original_signals_df['timestamp']).dt.strftime('%Y-%m-%d')
        
        price_columns = ['instrument_token', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.price_df = original_signals_df[price_columns].drop_duplicates()
        
        print(f"Price data prepared: {len(self.price_df)} daily price records")
    
    def get_next_day_data(self, instrument_token: int, signal_date: str) -> Optional[Dict]:
        """
        Get next day's price data for trade entry.
        
        Args:
            instrument_token: Instrument identifier
            signal_date: Date when signal was generated
            
        Returns:
            Dictionary with next day's OHLC data or None if not available
        """
        # Convert signal_date to datetime and add 1 day
        signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
        next_day_dt = signal_dt + timedelta(days=1)
        next_day_str = next_day_dt.strftime('%Y-%m-%d')
        
        # Find next day's data for this instrument
        next_day_data = self.price_df[
            (self.price_df['instrument_token'] == instrument_token) & 
            (self.price_df['timestamp'] == next_day_str)
        ]
        
        if next_day_data.empty:
            # Try to find the next available trading day within a reasonable window
            for i in range(1, 8):  # Check up to 7 days ahead (to handle weekends/holidays)
                check_date = (signal_dt + timedelta(days=i)).strftime('%Y-%m-%d')
                check_data = self.price_df[
                    (self.price_df['instrument_token'] == instrument_token) & 
                    (self.price_df['timestamp'] == check_date)
                ]
                if not check_data.empty:
                    return check_data.iloc[0].to_dict()
            return None
        
        return next_day_data.iloc[0].to_dict()
    
    def get_current_day_data(self, instrument_token: int, current_date: str) -> Optional[Dict]:
        """
        Get current day's price data.
        
        Args:
            instrument_token: Instrument identifier
            current_date: Current date
            
        Returns:
            Dictionary with current day's OHLC data or None if not available
        """
        current_data = self.price_df[
            (self.price_df['instrument_token'] == instrument_token) & 
            (self.price_df['timestamp'] == current_date)
        ]
        
        if current_data.empty:
            return None
        
        return current_data.iloc[0].to_dict()
    
    def run_backtest(self):
        """
        Run the complete backtesting process using simple iteration approach.
        """
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
            active_trade: Optional[Trade] = None
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
                            active_trade = None
                
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
                        
                        # Get next day's data for entry
                        next_day_data = self.get_next_day_data(instrument_token, date)
                        
                        if next_day_data is not None:
                            # Enter new trade
                            active_trade = Trade(
                                instrument_token=instrument_token,
                                entry_date=next_day_data['timestamp'],  # Actual entry date
                                entry_price=next_day_data['open'],
                                target_price_pct=self.target_price_pct,
                                stop_loss_pct=self.stop_loss_pct,
                                holding_period=self.holding_period
                            )
                            trades_entered += 1
            
            # Close any remaining active trade at the end
            if active_trade is not None:
                # Find the last available price data for this instrument
                last_day_data = self.price_df[
                    self.price_df['instrument_token'] == instrument_token
                ].sort_values('timestamp').iloc[-1]
                
                active_trade.close_trade(
                    last_day_data['timestamp'], 
                    last_day_data['close'], 
                    'data_end'
                )
                self.completed_trades.append(active_trade)
        
        print(f"✅ Completed: {total_signals_processed} signals → {len(self.completed_trades)} trades")
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Convert completed trades to DataFrame.
        
        Returns:
            DataFrame with all trade details
        """
        if not self.completed_trades:
            return pd.DataFrame()
        
        trades_data = [trade.to_dict() for trade in self.completed_trades]
        return pd.DataFrame(trades_data)
    
    def export_trades(self, output_path: str):
        """
        Export completed trades to CSV file.
        
        Args:
            output_path: Path to save the trades CSV file
        """
        trades_df = self.get_trades_dataframe()
        trades_df.to_csv(output_path, index=False)
        print(f"\nTrades exported to: {output_path}")
        print(f"Total trades exported: {len(trades_df)}") 