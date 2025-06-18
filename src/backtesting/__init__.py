"""
Backtesting module for algorithmic trading strategies.
"""

from .trade import Trade
from .backtest_engine import BacktestEngine
from .performance import PerformanceCalculator

__all__ = ['Trade', 'BacktestEngine', 'PerformanceCalculator'] 