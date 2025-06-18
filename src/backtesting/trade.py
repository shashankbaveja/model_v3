"""
Trade class to represent individual trading positions.
"""

from datetime import datetime, timedelta
from typing import Optional


class Trade:
    """Represents a single trade with entry, exit conditions and PnL tracking."""
    
    def __init__(self, instrument_token: int, entry_date: Optional[str], entry_price: Optional[float], 
                 target_price_pct: float, stop_loss_pct: float, holding_period: int):
        """
        Initialize a new trade.
        
        Args:
            instrument_token: Stock identifier
            entry_date: Date when signal was generated (YYYY-MM-DD) or None if no execution
            entry_price: Price at which trade was entered or None if no execution
            target_price_pct: Target profit percentage
            stop_loss_pct: Stop loss percentage
            holding_period: Maximum holding period in days
        """
        self.instrument_token = instrument_token
        self.entry_date = entry_date
        self.entry_price = entry_price
        
        # Handle case where no execution was possible
        if entry_price is None or entry_date is None:
            self.quantity = 0
            self.invested_amount = 0.0
            self.target_price = None
            self.stop_loss_price = None
            self.max_holding_date = None
            self.execution_status = 'no_execution'
        else:
            # Calculate derived values for normal trades
            self.quantity = int(10000 / entry_price)  # Fixed ₹10,000 investment
            self.invested_amount = self.quantity * entry_price
            
            # Exit conditions
            self.target_price = entry_price * (1 + target_price_pct / 100)
            self.stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            
            # Calculate max holding date
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            max_holding_dt = entry_dt + timedelta(days=holding_period - 1)
            self.max_holding_date = max_holding_dt.strftime('%Y-%m-%d')
            self.execution_status = 'executed'
        
        # Exit details (to be filled when trade closes)
        self.exit_date: Optional[str] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None
        self.pnl: Optional[float] = None
        self.return_pct: Optional[float] = None
        self.holding_days: Optional[int] = None
        
        # Trade status
        self.is_active = True if self.execution_status == 'executed' else False
    
    def check_exit_condition(self, current_date: str, current_close: float) -> tuple[bool, str]:
        """
        Check if trade should be exited based on current price and date.
        
        Args:
            current_date: Current date (YYYY-MM-DD)
            current_close: Current close price
            
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
        """
        # Skip exit checks for trades that were never executed
        if self.execution_status == 'no_execution':
            return False, ''
        
        # Priority: Target > Stop Loss > Holding Period
        
        # 1. Check target hit
        if current_close >= self.target_price:
            return True, 'target_hit'
        
        # 2. Check stop loss
        if current_close <= self.stop_loss_price:
            return True, 'stop_loss'
        
        # 3. Check holding period
        if current_date >= self.max_holding_date:
            return True, 'holding_period'
        
        return False, ''
    
    def close_trade(self, exit_date: str, exit_price: float, exit_reason: str):
        """
        Close the trade and calculate PnL.
        
        Args:
            exit_date: Date when trade was exited
            exit_price: Price at which trade was exited
            exit_reason: Reason for exit ('target_hit', 'stop_loss', 'holding_period', 'data_end')
        """
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.is_active = False
        
        # Calculate PnL only for executed trades
        if self.execution_status == 'executed' and self.entry_price is not None:
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.return_pct = (exit_price - self.entry_price) / self.entry_price * 100
            
            # Calculate holding days
            entry_dt = datetime.strptime(self.entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            self.holding_days = (exit_dt - entry_dt).days
        else:
            # No execution means no PnL
            self.pnl = 0.0
            self.return_pct = 0.0
            self.holding_days = 0
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary for easy export."""
        return {
            'instrument_token': self.instrument_token,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'invested_amount': self.invested_amount,
            'target_price': self.target_price,
            'stop_loss_price': self.stop_loss_price,
            'max_holding_date': self.max_holding_date,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
            'holding_days': self.holding_days,
            'execution_status': self.execution_status,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        if self.execution_status == 'no_execution':
            return f"Trade({self.instrument_token}, NO_EXECUTION)"
        status = "ACTIVE" if self.is_active else f"CLOSED ({self.exit_reason})"
        return f"Trade({self.instrument_token}, {self.entry_date}, ₹{self.entry_price:.2f}, {status})" 