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

sys_details = system_initialization()
kite = kiteAPIs()

# holdings = sys_details.kite.holdings()
# positions = sys_details.kite.positions()
trades = kite.get_trades()




# orders = sys_details.kite.orders()

# orders_df = pd.DataFrame(orders)
# print(orders_df.columns)
# print(orders_df.head())


# Symbols to exclude
# exclude_symbols = ['IDEA', 'IDFCFIRSTB', 'YESBANK']

# # Combine and filter data
# combined_df = combine_and_filter_data(holdings_data, positions_data, exclude_symbols, config)

