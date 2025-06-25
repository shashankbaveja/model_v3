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

systemDetails = system_initialization()
systemDetails.run_query_limit(f"Call trades_PnL();")
Pnl = systemDetails.GetPnL()
Pnl.to_csv('todays_trades/pnl.csv')

print(Pnl)