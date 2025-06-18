"""
Performance calculation module for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from .trade import Trade


class PerformanceCalculator:
    """Calculate comprehensive performance metrics from completed trades."""
    
    def __init__(self, trades: List[Trade]):
        """
        Initialize with completed trades.
        
        Args:
            trades: List of completed Trade objects
        """
        self.trades = trades
        self.trades_df = self._trades_to_dataframe()
        
    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades list to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(trades_data)
        
        # Convert dates to datetime for calculations
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        
        return df
    
    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic trading performance metrics."""
        if self.trades_df.empty:
            return self._empty_metrics()
        
        df = self.trades_df
        
        # Basic counts
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        breakeven_trades = len(df[df['pnl'] == 0])
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = df['pnl'].sum()
        gross_profit = df[df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Average trade metrics
        avg_trade_pnl = df['pnl'].mean()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Return percentages
        avg_return_pct = df['return_pct'].mean()
        avg_win_pct = df[df['pnl'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_loss_pct = df[df['pnl'] < 0]['return_pct'].mean() if losing_trades > 0 else 0
        
        # Best and worst trades
        best_trade_pnl = df['pnl'].max()
        worst_trade_pnl = df['pnl'].min()
        best_trade_pct = df['return_pct'].max()
        worst_trade_pct = df['return_pct'].min()
        
        # Holding period statistics
        avg_holding_days = df['holding_days'].mean()
        min_holding_days = df['holding_days'].min()
        max_holding_days = df['holding_days'].max()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate_pct': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 3),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_win_pnl': round(avg_win, 2),
            'avg_loss_pnl': round(avg_loss, 2),
            'avg_return_pct': round(avg_return_pct, 3),
            'avg_win_pct': round(avg_win_pct, 3),
            'avg_loss_pct': round(avg_loss_pct, 3),
            'best_trade_pnl': round(best_trade_pnl, 2),
            'worst_trade_pnl': round(worst_trade_pnl, 2),
            'best_trade_pct': round(best_trade_pct, 3),
            'worst_trade_pct': round(worst_trade_pct, 3),
            'avg_holding_days': round(avg_holding_days, 2),
            'min_holding_days': min_holding_days,
            'max_holding_days': max_holding_days
        }
    
    def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown and risk metrics."""
        if self.trades_df.empty:
            return {'max_drawdown_pct': 0, 'max_drawdown_pnl': 0, 'drawdown_duration_days': 0}
        
        df = self.trades_df.copy()
        df = df.sort_values('exit_date')
        
        # Calculate cumulative PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Calculate running maximum (peak)
        df['running_max'] = df['cumulative_pnl'].expanding().max()
        
        # Calculate drawdown
        df['drawdown'] = df['cumulative_pnl'] - df['running_max']
        
        # Maximum drawdown
        max_drawdown_pnl = df['drawdown'].min()  # Most negative value
        
        # Calculate max drawdown percentage (relative to peak)
        max_dd_idx = df['drawdown'].idxmin()
        peak_value = df.loc[max_dd_idx, 'running_max']
        max_drawdown_pct = (max_drawdown_pnl / peak_value * 100) if peak_value != 0 else 0
        
        # Drawdown duration (simplified - time between peak and recovery)
        drawdown_duration_days = 0
        if not df.empty:
            # Find periods of drawdown
            in_drawdown = df['drawdown'] < 0
            if in_drawdown.any():
                # Find the longest consecutive drawdown period
                drawdown_periods = []
                current_start = None
                
                for idx, is_dd in in_drawdown.items():
                    if is_dd and current_start is None:
                        current_start = idx
                    elif not is_dd and current_start is not None:
                        period_length = (df.loc[idx, 'exit_date'] - df.loc[current_start, 'exit_date']).days
                        drawdown_periods.append(period_length)
                        current_start = None
                
                # Handle case where drawdown continues to the end
                if current_start is not None and len(df) > 1:
                    period_length = (df['exit_date'].iloc[-1] - df.loc[current_start, 'exit_date']).days
                    drawdown_periods.append(period_length)
                
                drawdown_duration_days = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown_pnl': round(max_drawdown_pnl, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'drawdown_duration_days': drawdown_duration_days
        }
    
    def calculate_exit_reason_breakdown(self) -> Dict[str, Any]:
        """Calculate breakdown by exit reasons."""
        if self.trades_df.empty:
            return {}
        
        exit_counts = self.trades_df['exit_reason'].value_counts()
        exit_percentages = (exit_counts / len(self.trades_df) * 100).round(2)
        
        # PnL by exit reason
        exit_pnl = self.trades_df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']).round(2)
        
        breakdown = {}
        for reason in exit_counts.index:
            breakdown[f'{reason}_count'] = exit_counts[reason]
            breakdown[f'{reason}_pct'] = exit_percentages[reason]
            breakdown[f'{reason}_total_pnl'] = exit_pnl.loc[reason, 'sum']
            breakdown[f'{reason}_avg_pnl'] = exit_pnl.loc[reason, 'mean']
        
        return breakdown
    
    def calculate_monthly_performance(self) -> pd.DataFrame:
        """Calculate monthly performance breakdown."""
        if self.trades_df.empty:
            return pd.DataFrame()
        
        df = self.trades_df.copy()
        
        # Extract year-month from exit date
        df['year_month'] = df['exit_date'].dt.to_period('M')
        
        # Calculate monthly statistics
        monthly_stats = df.groupby('year_month').agg({
            'pnl': ['count', 'sum', 'mean'],
            'return_pct': 'mean',
            'holding_days': 'mean'
        }).round(2)
        
        # Flatten column names
        monthly_stats.columns = ['trades_count', 'total_pnl', 'avg_pnl', 'avg_return_pct', 'avg_holding_days']
        
        # Calculate win rate per month
        monthly_wins = df[df['pnl'] > 0].groupby('year_month').size()
        monthly_stats['win_rate_pct'] = (monthly_wins / monthly_stats['trades_count'] * 100).fillna(0).round(2)
        
        # Reset index to make year_month a column
        monthly_stats = monthly_stats.reset_index()
        monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)
        
        return monthly_stats
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all performance metrics."""
        basic_metrics = self.calculate_basic_metrics()
        drawdown_metrics = self.calculate_drawdown_metrics()
        exit_breakdown = self.calculate_exit_reason_breakdown()
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **drawdown_metrics, **exit_breakdown}
        
        return all_metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when no trades available."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate_pct': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'profit_factor': 0,
            'avg_trade_pnl': 0,
            'avg_win_pnl': 0,
            'avg_loss_pnl': 0,
            'avg_return_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'best_trade_pnl': 0,
            'worst_trade_pnl': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0,
            'avg_holding_days': 0,
            'min_holding_days': 0,
            'max_holding_days': 0
        }
    
    def print_performance_summary(self):
        """Print a formatted performance summary."""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*80)
        print("BACKTESTING PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\n📊 BASIC METRICS:")
        print(f"  Total Trades: {metrics['total_trades']:,}")
        print(f"  Winning Trades: {metrics['winning_trades']:,} ({metrics['win_rate_pct']:.2f}%)")
        print(f"  Losing Trades: {metrics['losing_trades']:,}")
        print(f"  Breakeven Trades: {metrics['breakeven_trades']:,}")
        
        print(f"\n💰 PnL METRICS:")
        print(f"  Total PnL: ₹{metrics['total_pnl']:,.2f}")
        print(f"  Gross Profit: ₹{metrics['gross_profit']:,.2f}")
        print(f"  Gross Loss: ₹{metrics['gross_loss']:,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        
        print(f"\n📈 AVERAGE TRADE:")
        print(f"  Average PnL: ₹{metrics['avg_trade_pnl']:,.2f}")
        print(f"  Average Win PnL: ₹{metrics['avg_win_pnl']:,.2f}")
        print(f"  Average Loss PnL: ₹{metrics['avg_loss_pnl']:,.2f}")
        print(f"  Average Return: {metrics['avg_return_pct']:.3f}%")
        print(f"  Average Holding: {metrics['avg_holding_days']:.1f} days")
        
        print(f"\n🎯 BEST/WORST:")
        print(f"  Best Trade: ₹{metrics['best_trade_pnl']:,.2f} ({metrics['best_trade_pct']:.2f}%)")
        print(f"  Worst Trade: ₹{metrics['worst_trade_pnl']:,.2f} ({metrics['worst_trade_pct']:.2f}%)")
        
        print(f"\n📉 RISK METRICS:")
        print(f"  Max Drawdown: ₹{metrics['max_drawdown_pnl']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Drawdown Duration: {metrics['drawdown_duration_days']} days")
        
        # Exit reason breakdown
        exit_reasons = ['target_hit', 'stop_loss', 'holding_period', 'data_end']
        print(f"\n🚪 EXIT REASONS:")
        for reason in exit_reasons:
            count_key = f'{reason}_count'
            pct_key = f'{reason}_pct'
            pnl_key = f'{reason}_total_pnl'
            if count_key in metrics:
                print(f"  {reason.replace('_', ' ').title()}: {metrics[count_key]} trades ({metrics[pct_key]:.1f}%) | PnL: ₹{metrics[pnl_key]:,.2f}")
        
        print("="*80)
    
    def export_performance_report(self, output_path: str):
        """Export detailed performance report to CSV."""
        metrics = self.calculate_all_metrics()
        monthly_perf = self.calculate_monthly_performance()
        
        # Save overall metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = output_path.replace('.csv', '_summary.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save monthly performance
        if not monthly_perf.empty:
            monthly_path = output_path.replace('.csv', '_monthly.csv')
            monthly_perf.to_csv(monthly_path, index=False)
        
        print(f"\nPerformance report exported:")
        print(f"  Summary: {metrics_path}")
        if not monthly_perf.empty:
            print(f"  Monthly: {monthly_path}") 