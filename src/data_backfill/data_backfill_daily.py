# /opt/anaconda3/bin/activate && conda activate /opt/anaconda3/envs/KiteConnect

# 0 16 * * * cd /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup && /opt/anaconda3/envs/KiteConnect/bin/python data_backfill.py >> /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup/data_backfill_cron.log 2>&1

# Add project root to sys.path to resolve module imports
import sys
import os
from IPython import embed;
from kiteconnect import KiteConnect, KiteTicker
import mysql
import mysql.connector as sqlConnector
import datetime
from selenium import webdriver
import os
from pyotp import TOTP
import ast
import time
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from myKiteLib import system_initialization, kiteAPIs
import logging
import json
from datetime import date, timedelta, datetime, time
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions
import requests # Import requests for ReadTimeout

from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    BACKFILL_INTERVAL = 'day'
    BACKFILL_DAYS = 1
    
    today_date = date.today()
    
    end_dt_backfill = datetime.combine(today_date, time(23, 59, 59))
    
    start_date_val = today_date - timedelta(days=BACKFILL_DAYS)
    start_dt_backfill = datetime.combine(start_date_val, time(0, 0, 0))

    print(f"starting system_init")
    systemDetails = system_initialization()
    systemDetails.init_trading()
    callKite = kiteAPIs()
    trades = callKite.get_trades()
    print("Trades Updated in DB!")
    tokenList = [256265] ## NIFTY INDEX
    tokenList.extend(callKite.get_instrument_all_tokens('EQ'))


    try:
        print(f"Deleting data for {start_dt_backfill} to {end_dt_backfill}")
        result = systemDetails.DeleteData("Delete from kiteconnect.historical_data_day where timestamp >= DATE_FORMAT(CURDATE() - 1, '%Y-%m-%d')")
        print(f"Result: {result}")
        df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)
    except (KiteException, requests.exceptions.ReadTimeout) as e:
        print(f"Error fetching historical data: {e}")
        logging.error(f"Error fetching historical data: {e}")
        df = pd.DataFrame() # Initialize an empty DataFrame or handle as needed
    systemDetails.run_query_limit(f"Call trades_PnL();")
    Pnl = systemDetails.GetPnL()
    Pnl.to_csv('todays_trades/pnl.csv')