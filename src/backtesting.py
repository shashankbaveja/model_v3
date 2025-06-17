import os
import sys
import pandas as pd
import numpy as np
import yaml
import re
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config


class Backtester:
    """
    An event-driven backtester that simulates trading strategies on historical data.

    This class is designed to handle portfolio-level backtesting with realistic
    capital management, position sizing, and transaction costs.
    """
    
    def __init__(self, initial_capital, signals_df, price_data_grouped, trade_direction, 
                 sl_pct, tp_pct, holding_period_days, txn_cost_pct):
        """
        Initializes the Backtester.

        Args:
            initial_capital (float): The starting capital for the backtest.
            signals_df (pd.DataFrame): DataFrame containing trading signals. 
                                       Must have a MultiIndex of ('instrument_token', 'timestamp').
            price_data_grouped (pd.groupby): Price data grouped by timestamp for efficient access.
            trade_direction (str): 'up' for long trades, 'down' for short trades.
            sl_pct (float): Stop loss percentage (e.g., 1.0 for 1%).
            tp_pct (float): Take profit percentage (e.g., 3.0 for 3%).
            holding_period_days (int): Maximum number of days to hold a position.
            txn_cost_pct (float): Transaction cost percentage per trade.
        """
        self.initial_capital = float(initial_capital)
        self.signals_df = signals_df
        self.price_data_grouped = price_data_grouped
        self.trade_direction = trade_direction
        self.sl_pct = sl_pct / 100.0
        self.tp_pct = tp_pct / 100.0
        self.holding_period = pd.to_timedelta(holding_period_days, unit='D')
        self.txn_cost_pct = txn_cost_pct / 100.0

        # State variables
        self.cash = self.initial_capital
        self.equity_curve = pd.Series(dtype=float)
        self.open_positions = {}
        self.closed_trades = []

    def run(self):
        """Executes the backtest simulation from start to finish."""
        print(f"--- Running Backtest: Direction='{self.trade_direction.upper()}', SL={self.sl_pct*100:.1f}%, TP={self.tp_pct*100:.1f}% ---")
        
        for timestamp, candle_group in self.price_data_grouped:
            self._update_equity(timestamp)
            
            # Order of operations is important: exit checks before entry checks
            self._check_exits(timestamp, candle_group)
            self._check_entries(timestamp, candle_group)

        self._mark_to_market_at_close()
        print("--- Backtest Complete ---")
        
    def _check_exits(self, timestamp, candle_group):
        """Checks and processes exits for all open positions."""
        # Iterate over a copy of keys as the dictionary can be modified
        for instrument_token in list(self.open_positions.keys()):
            pos = self.open_positions[instrument_token]
            exit_reason = None
            exit_price = 0

            # If instrument data is not available at this timestamp, hold the position.
            if instrument_token not in candle_group.index.get_level_values('instrument_token'):
                continue

            candle = candle_group.loc[instrument_token]

            # Determine exit conditions based on trade direction
            if self.trade_direction == 'up':
                if candle['low'].item() <= pos['stop_loss']:
                    exit_price, exit_reason = pos['stop_loss'], 'Stop Loss'
                elif candle['high'].item() >= pos['take_profit']:
                    exit_price, exit_reason = pos['take_profit'], 'Take Profit'
            else: # 'down'
                if candle['high'].item() >= pos['stop_loss']:
                    exit_price, exit_reason = pos['stop_loss'], 'Stop Loss'
                elif candle['low'].item() <= pos['take_profit']:
                    exit_price, exit_reason = pos['take_profit'], 'Take Profit'
            
            # Time-based exit
            if not exit_reason and timestamp >= pos['exit_by_timestamp']:
                exit_price, exit_reason = candle['close'].item(), 'Holding Period'
            
            if exit_reason:
                self._close_position(instrument_token, timestamp, exit_price, exit_reason)

    def _check_entries(self, timestamp, candle_group):
        """Checks for new signals and enters new positions."""
        # Find all signals for the current timestamp
        current_signals = self.signals_df.index.intersection(candle_group.index)
        
        new_entries = []
        for instrument_token, ts in current_signals:
            # Ensure we don't enter a position for an instrument we already hold
            if instrument_token not in self.open_positions:
                new_entries.append(instrument_token)
        
        if not new_entries:
            return

        # Capital allocation: Divide available cash equally among all new entries
        capital_per_trade = self.cash / len(new_entries)
        
        for instrument_token in new_entries:
            entry_price = candle_group.loc[instrument_token]['close'].item()
            
            # Calculate SL and TP prices
            if self.trade_direction == 'up':
                stop_loss = entry_price * (1 - self.sl_pct)
                take_profit = entry_price * (1 + self.tp_pct)
            else: # 'down'
                stop_loss = entry_price * (1 + self.sl_pct)
                take_profit = entry_price * (1 - self.tp_pct)

            # Deduct allocated capital from cash
            self.cash -= capital_per_trade

            # Record the new position
            self.open_positions[instrument_token] = {
                'entry_timestamp': timestamp,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'exit_by_timestamp': timestamp + self.holding_period,
                'capital_allocated': capital_per_trade
            }

    def _close_position(self, instrument_token, exit_timestamp, exit_price, exit_reason):
        """Closes a position and records the trade details."""
        pos = self.open_positions.pop(instrument_token)
        
        # Calculate PnL
        entry_price = pos['entry_price']
        capital_allocated = pos['capital_allocated']
        
        if self.trade_direction == 'up':
            pnl_pct = (exit_price - entry_price) / entry_price
        else: # 'down'
            pnl_pct = (entry_price - exit_price) / entry_price
        
        gross_pnl = pnl_pct * capital_allocated
        txn_cost = abs(gross_pnl * self.txn_cost_pct)
        net_pnl = gross_pnl - txn_cost
        
        # Return capital to cash pool
        self.cash += (capital_allocated + net_pnl)

        self.closed_trades.append({
            'instrument_token': instrument_token,
            'entry_timestamp': pos['entry_timestamp'],
            'exit_timestamp': exit_timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason
        })

    def _update_equity(self, timestamp):
        """Updates the daily equity value of the portfolio."""
        # In a more complex model, we would mark-to-market open positions here.
        # For this model, we track equity based on cash + allocated capital.
        total_equity = self.cash + sum(p['capital_allocated'] for p in self.open_positions.values())
        self.equity_curve[timestamp] = total_equity
        
    def _mark_to_market_at_close(self):
        """Closes all remaining open positions at the end of the backtest period."""
        print(f"  Marking {len(self.open_positions)} open positions to market at close...")
        if not self.open_positions:
            return
            
        try:
            # Get the very last timestamp from the grouped data iterator
            last_timestamp = max(self.price_data_grouped.groups.keys())
            last_day_prices = self.price_data_grouped.get_group(last_timestamp)
            last_day_prices = last_day_prices.droplevel('timestamp')

            for token, pos in list(self.open_positions.items()):
                if token in last_day_prices.index:
                    exit_price = last_day_prices.loc[token]['close']
                    self._close_position(token, last_timestamp, exit_price, 'End of Backtest')
                else:
                    # If no price on the last day, close at entry price (zero PnL)
                    self._close_position(token, pos['entry_timestamp'], pos['entry_price'], 'End of Backtest (No Final Price)')
        except Exception as e:
            print(f"Could not mark-to-market. Error: {e}")


    def get_results(self):
        """Returns the results of the backtest."""
        return pd.DataFrame(self.closed_trades), self.equity_curve


def analyze_performance(trades_df, equity_curve, initial_capital):
    """Analyzes and prints the performance of the backtest."""
    print("\n--- Backtest Performance Analysis ---")
    
    results = {
        'Total Return Pct': 0, 'Final Equity': initial_capital, 'Total Trades': 0,
        'Win Rate Pct': 0, 'Profit Factor': 'N/A', 'Sharpe Ratio': 0, 'Max Drawdown Pct': 0
    }

    if trades_df.empty:
        print("No trades were executed.")
        return results

    closed_trades_df = trades_df[pd.to_numeric(trades_df['net_pnl'], errors='coerce').notna()].copy()
    if closed_trades_df.empty:
        print("No closed trades with numerical PnL to analyze.")
        results['Total Trades'] = len(trades_df)
        return results
    
    total_trades = len(closed_trades_df)
    total_pnl = closed_trades_df['net_pnl'].sum()
    final_equity = initial_capital + total_pnl
    total_return_pct = (total_pnl / initial_capital) * 100
    
    winning_trades = closed_trades_df[closed_trades_df['net_pnl'] > 0]
    losing_trades = closed_trades_df[closed_trades_df['net_pnl'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = winning_trades['net_pnl'].sum()
    gross_loss = abs(losing_trades['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0 # Annualized
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100

    results.update({
        'Total Return Pct': total_return_pct, 'Final Equity': final_equity, 'Total Trades': total_trades,
        'Win Rate Pct': win_rate, 'Profit Factor': f"{profit_factor:.2f}", 
        'Sharpe Ratio': sharpe_ratio, 'Max Drawdown Pct': max_drawdown
    })

    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print("\nExit Reasons:\n" + str(closed_trades_df['exit_reason'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}")))
    
    return results


def main():
    """Main function to run the entire backtesting suite."""
    print("--- Starting Backtest Suite ---")
    config = load_config()

    # --- Load Parameters ---
    sim_params = config.get('trading', {}).get('simulation_params', {})
    target_params = config.get('target_generation', {})
    
    # initial_capital = 500000 # As requested
    initial_capital = sim_params.get('initial_capital', 500000)
    txn_cost_pct = sim_params.get('transaction_cost_pct', 0.1)
    sl_pct = target_params.get('stop_loss_pct', 1.0)
    tp_pct = target_params.get('threshold_percent', 3.0)
    holding_period = target_params.get('lookahead_candles', 5)

    # --- Prepare Price Data ---
    try:
        price_data = pd.read_parquet('data/processed/test_raw.parquet')
        price_data.set_index(['instrument_token', 'timestamp'], inplace=True)
        price_data.sort_index(inplace=True)
    except FileNotFoundError:
        print("Error: 'data/processed/test_raw.parquet' not found. Run data pipeline.", file=sys.stderr)
        sys.exit(1)
    
    print("Pre-grouping price data by timestamp for performance...")
    grouped_price_data = price_data.groupby(level='timestamp', sort=True)
    print("Pre-grouping complete.")

    # --- Setup Directories ---
    signals_dir = 'data/signals'
    trade_logs_dir = 'reports/trade_logs'
    os.makedirs(trade_logs_dir, exist_ok=True)

    if not os.path.isdir(signals_dir) or not os.listdir(signals_dir):
        print(f"Error: Signal directory '{signals_dir}' is empty or not found.", file=sys.stderr)
        sys.exit(1)

    all_results = []

    # --- Loop Through Signal Files ---
    for filename in sorted(os.listdir(signals_dir)):
        if not filename.endswith('_signals.parquet'):
            continue

        model_id = filename.replace('_signals.parquet', '')
        print("\n" + "="*70)
        print(f"Processing Model: {model_id}")
        print("="*70)

        signal_path = os.path.join(signals_dir, filename)
        signal_data_df = pd.read_parquet(signal_path)
        
        # Prepare signals for the backtester
        signals_df = signal_data_df[signal_data_df['signal'] == 1].copy()
        if signals_df.empty:
            print(f"No signals found for {model_id}. Skipping.")
            continue
        signals_df.set_index(['instrument_token', 'timestamp'], inplace=True)
        
        trade_direction = 'down' if '_down_' in model_id else 'up'

        # --- Initialize and Run Backtester ---
        backtester = Backtester(
            initial_capital=initial_capital,
            signals_df=signals_df,
            price_data_grouped=grouped_price_data,
            trade_direction=trade_direction,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            holding_period_days=holding_period,
            txn_cost_pct=txn_cost_pct
        )
        backtester.run()

        # --- Analyze and Save Results ---
        trades, equity = backtester.get_results()

        if not trades.empty:
            log_path = os.path.join(trade_logs_dir, f"{model_id}_trades.csv")
            trades.to_csv(log_path, index=False)
            print(f"  Trade log saved to: {log_path}")

        perf_summary = analyze_performance(trades, equity, initial_capital)
        
        match = re.search(r'^(.*)_thresh_(\d+\.\d+)', model_id)
        if match:
            perf_summary['Model Name'] = match.group(1)
            perf_summary['Threshold'] = float(match.group(2))
        else:
            perf_summary['Model Name'] = model_id
            perf_summary['Threshold'] = 'N/A'
            
        all_results.append(perf_summary)

    # --- Save Consolidated Report ---
    if all_results:
        pnl_df = pd.DataFrame(all_results)
        report_path = os.path.join('reports', 'pnl_summary.csv')
        pnl_df.to_csv(report_path, index=False)
        print(f"\nConsolidated PnL report saved to: {report_path}")
    else:
        print("\nNo backtests were run. No summary report generated.")
        
    print("\n--- Backtest Suite Finished ---")


if __name__ == "__main__":
    main() 