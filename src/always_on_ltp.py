import time
import datetime
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the Python path to allow importing myKiteLib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import system_initialization, OrderPlacement, kiteAPIs
from src.data_pipeline import load_config

# 1. Placeholder functions for user to implement

def get_trade_dates(sys_init):
    query = "Select tradingsymbol, max(fill_timestamp) as trade_date_db from kiteconnect.trades where transaction_type = 'BUY' and fill_timestamp >= curdate() - interval 8 day group by 1;"
    df = sys_init.run_query_full(query)
    return df

def get_holdings(callKite, sys_init, order_placement):
    print("INFO: Fetching holdings...")
    holdings_data = callKite.extract_holdings_data()
    positions_data = callKite.extract_positions_data() 
    all_data = holdings_data + positions_data

    print(all_data)
    df = pd.DataFrame(all_data)
    dates = get_trade_dates(sys_init)

    exclude_symbols = ['IDEA', 'IDFCFIRSTB', 'YESBANK']
    if not df.empty:
        df = df[~df['tradingsymbol'].isin(exclude_symbols)]
    
    merged_df = pd.merge(df, dates, on='tradingsymbol', how='left')
    merged_df['trade_date_db'] = pd.to_datetime(merged_df['trade_date_db'])

    condition = merged_df['data_source'] == 'positions'
    holding_days = (pd.Timestamp.now().normalize() - merged_df['trade_date_db']).dt.days
    merged_df['holding_period'] = np.where(
        condition,       # If data_source is 'positions'...
        0,               # ...then set holding_period to 0.
        holding_days     # Otherwise, set it to the calculated holding days.
    )

    token_list = merged_df['instrument_token'].unique().tolist()
    ltp_dict = order_placement.get_ltp_live(token_list)
    print(token_list)
    print(ltp_dict)
    merged_df['ltp'] = merged_df['instrument_token'].map(ltp_dict)
    merged_df['pnl'] = merged_df['quantity']*(merged_df['ltp'] - merged_df['average_price'])
    merged_df['pnl_percent'] = (100*merged_df['ltp']/merged_df['average_price'])-100
    merged_df = merged_df.drop(columns=['trade_date', 'last_price'])
    return merged_df

def exit_trade(order_placement_instance, position, reason):
    tradingsymbol = position['tradingsymbol']
    exchange = position['exchange']
    quantity = position['quantity']  
    print(f"INFO: Exiting trade for {tradingsymbol} ({quantity} qty) due to: {reason}")
  
    order_placement_instance.place_market_order_live(tradingsymbol, exchange, 'SELL', quantity, 'CNC', reason)
    order_placement_instance.send_telegram_message(f"Exiting trade for {tradingsymbol} ({quantity} qty) due to: {reason}")
    print(f"SIMULATION: Exit order for {tradingsymbol} would be placed here.")
    
def main_monitoring_loop():
    """
    The main loop that runs continuously to initialize the session and monitor trades.
    """
    print("Initializing Kite session...")
    order_placement = None
    callKite = None
    config = load_config()
    holding_period = config['target_generation']['lookahead_candles']
    target_pct = config['target_generation']['threshold_percent']
    stop_loss_pct = -config['target_generation']['stop_loss_pct']

    print(f"holding_period: {holding_period}, target_pct: {target_pct}, stop_loss_pct: {stop_loss_pct}")

    try:
        sys_init = system_initialization()
        sys_init.init_trading()
        order_placement = OrderPlacement()
        order_placement.init_trading()
        callKite = kiteAPIs()
        trades = callKite.get_trades()
        sys_init.run_query_limit(f"Call trades_PnL();")
    except Exception as e:
        print(f"FATAL: Failed to initialize Kite session: {e}")
        return

    monitored_today = set() # To track symbols exited today to avoid re-triggering

    while True:
        try:
            # Check if current time is within trading hours
            now = datetime.datetime.now()
            market_start_time = now.replace(hour=8, minute=15, second=0, microsecond=0)
            market_end_time = now.replace(hour=15, minute=29, second=0, microsecond=0)

            if market_start_time <= now <= market_end_time:
                print("\n--- Market Open: Running Monitoring Cycle ---")
                
                active_trades = get_holdings(callKite, sys_init, order_placement)
                print(active_trades)
                print("Total PnL: ", sum(active_trades['PnL']))
                if active_trades.empty:
                    print("INFO: No holdings to monitor at this time.")
                else:
                    for position in active_trades.to_dict('records'):
                        # print("Trading Symbol: ", position['tradingsymbol'], "Days Diff: ", days_diff, "Trade Date: ", trade_date)
                        if position['quantity'] != 0:
                            if position['pnl_percent'] > target_pct:
                                exit_trade(order_placement, position, 'TP Hit')
                                monitored_today.add(position['tradingsymbol'])
                            if position['pnl_percent'] < stop_loss_pct:
                                exit_trade(order_placement, position, 'SL Hit')
                                monitored_today.add(position['tradingsymbol'])
                            if position['holding_period'] > 7 and now == now.replace(hour=15, minute=15, second=0, microsecond=0):
                                exit_trade(order_placement, position, 'HP Hit')
                                monitored_today.add(position['tradingsymbol'])
                    
                print("--- Cycle Finished. Sleeping for 1 minute. ---")
                time.sleep(60) 

            elif now.replace(second=0, microsecond=0) == now.replace(hour=15, minute=30, second=0, microsecond=0):
                systemDetails.run_query_limit(f"Call trades_PnL();")
                Pnl = systemDetails.GetPnL()
                Pnl.to_csv('todays_trades/pnl.csv')
                
            else:
                # Reset the set of monitored symbols outside of market hours
                print("entered Else loop")
                if monitored_today:
                    print(f"INFO: Market closed. Resetting exited symbols for next session: {monitored_today}")
                    monitored_today.clear()
                
                print(f"INFO: Market Closed. Current time: {now.strftime('%H:%M:%S')}. Sleeping for 5 minutes.")
                time.sleep(300) # Sleep longer outside market hours

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            # if order_placement:
            #     order_placement.send_telegram_message("Always-on LTP monitor stopped manually.")
            break
        except Exception as e:
            print(f"ERROR: An unexpected error occurred in the main loop: {e}")
            if order_placement:
                order_placement.send_telegram_message(f"ERROR in LTP monitor: {e}")
            print("Waiting for 10 seconds before retrying...")
            time.sleep(10)

if __name__ == "__main__":
    main_monitoring_loop() 
    