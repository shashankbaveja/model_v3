import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import yaml
import warnings
import json

sys.tracebacklimit = 0
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import kiteAPIs, system_initialization, OrderPlacement


def load_config():
    """Load configuration parameters from parameters.yml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'parameters.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_holdings_data(holdings_json):
    """Extract required fields from holdings JSON data."""
    holdings_data = []
    for holding in holdings_json:
        # Get authorised_date and subtract one day to get actual trade_date
        auth_date_str = holding.get('authorised_date')
        if auth_date_str:
            auth_date = datetime.strptime(auth_date_str, '%Y-%m-%d %H:%M:%S')
            trade_date = auth_date - timedelta(days=1)
            trade_date_str = trade_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            trade_date_str = None
            
        holdings_data.append({
            'tradingsymbol': holding.get('tradingsymbol'),
            'instrument_token': holding.get('instrument_token'),
            'quantity': holding.get('quantity'),
            'trade_date': trade_date_str,
            'average_price': holding.get('average_price'),
            'last_price': holding.get('last_price'),
            'pnl': holding.get('pnl'),
            'data_source': 'holdings'
        })
    return holdings_data

def extract_positions_data(positions_json):
    """Extract required fields from positions JSON data (net positions)."""
    positions_data = []
    # For positions, use current date minus 1 day as trade_date
    current_date = datetime.now()
    trade_date_str = current_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract from 'net' positions
    net_positions = positions_json.get('net', [])
    for position in net_positions:
        positions_data.append({
            'tradingsymbol': position.get('tradingsymbol'),
            'instrument_token': position.get('instrument_token'),
            'quantity': position.get('quantity'),
            'trade_date': trade_date_str,
            'average_price': position.get('average_price'),
            'last_price': position.get('last_price'),
            'pnl': position.get('pnl'),
            'data_source': 'positions'
        })
    return positions_data

def calculate_trading_parameters(df, config):
    """Calculate TP, SL, and holding period based on config parameters."""
    if df.empty:
        return df
    
    # Get parameters from config
    target_price_pct = config['backtest']['target_price_pct']
    stop_loss_pct = config['backtest']['stop_loss_pct']
    holding_period = config['backtest']['holding_period']
    
    # Calculate target price (TP) and stop loss (SL)
    df['target_price'] = df['average_price'] * (1 + target_price_pct / 100)
    df['stop_loss'] = df['average_price'] * (1 - stop_loss_pct / 100)
    df['pnl_pct'] = 100*((df['last_price']/df['average_price']) - 1)
    
    # Calculate holding period end date
    df['holding_end_date'] = pd.to_datetime(df['trade_date']) + timedelta(days=holding_period)
    df['holding_end_date'] = df['holding_end_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    current_date = datetime.now()
    df['holding_days'] = pd.to_datetime(current_date.strftime('%Y-%m-%d %H:%M:%S')) - pd.to_datetime(df['trade_date'])
    df['holding_days'] = df['holding_days'].dt.days + 1
    
    # Add the parameter values as columns for reference
    df['target_pct'] = target_price_pct
    df['stop_loss_pct'] = stop_loss_pct
    df['holding_period_days'] = holding_period
    
    return df

def combine_and_filter_data(holdings_data, positions_data, exclude_symbols, config):
    """Combine holdings and positions data, then filter out specified symbols and add trading parameters."""
    # Combine both datasets
    all_data = holdings_data + positions_data
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Filter out specified trading symbols
    if not df.empty:
        df = df[~df['tradingsymbol'].isin(exclude_symbols)]
        
        # Calculate trading parameters
        df = calculate_trading_parameters(df, config)
    
    return df

# Load configuration
config = load_config()
sys_details = system_initialization()
sys_details.init_trading()

holdings = sys_details.kite.holdings()
positions = sys_details.kite.positions()

print("Holdings")
print(holdings)
print("Positions")
print(positions)

# Parse and combine the data
print("\n" + "="*50)
print("PROCESSED DATA")
print("="*50)

# Extract data from JSON
holdings_data = extract_holdings_data(holdings)
positions_data = extract_positions_data(positions)

# Symbols to exclude
exclude_symbols = ['IDEA', 'IDFCFIRSTB', 'YESBANK']

# Combine and filter data
combined_df = combine_and_filter_data(holdings_data, positions_data, exclude_symbols, config)

print(f"\nCombined and Filtered DataFrame with Trading Parameters:")
print(f"Total records: {len(combined_df)}")

if not combined_df.empty:
    # Display configuration parameters used
    print(f"\nTrading Parameters from config:")
    print(f"Target Price %: {config['backtest']['target_price_pct']}%")
    print(f"Stop Loss %: {config['backtest']['stop_loss_pct']}%")
    print(f"Holding Period: {config['backtest']['holding_period']} days")
    
    print("\nDataFrame:")
    # Select columns for better display
    display_columns = ['tradingsymbol', 'instrument_token', 'quantity', 'trade_date', 
                      'average_price', 'last_price', 'target_price', 'stop_loss', 
                      'holding_days', 'pnl', 'pnl_pct']
    print(combined_df[display_columns].to_string(index=False))
    
    print(f"\nTotal PnL: {combined_df['pnl'].sum():.2f}")
    print(f"Average Target Price: {combined_df['target_price'].mean():.2f}")
    print(f"Average Stop Loss: {combined_df['stop_loss'].mean():.2f}")

    combined_df.to_csv('todays_trades/open_positions.csv', header=True, index=False)
else:
    print("No data available after filtering.")


telegram_df = combined_df[['tradingsymbol', 'quantity', 'trade_date', 'average_price', 'last_price', 'holding_days', 'pnl', 'pnl_pct']]
order_placement = OrderPlacement()
order_placement.send_telegram_message("Existing Trades:")
order_placement.send_telegram_message(telegram_df.to_json(orient='records', indent=4))


# Clean up database connections to avoid MySQL cleanup warnings
try:
    if hasattr(sys_details, 'db_connection'):
        sys_details.close_db_connection()
        del sys_details
except:
    pass