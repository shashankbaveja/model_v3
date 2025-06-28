import json
from kiteconnect import KiteConnect, KiteTicker
import mysql
import numpy as np
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
import requests
from IPython import embed
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
import warnings
# from datetime import datetime as dt_datetime

# Test 2
# Suppress the specific pandas UserWarning about using non-SQLAlchemy connections
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

telegramToken_global = '8135376207:AAFoMWbyucyPPEzc7CYeAMTsNZfqHWYDMfw' # Renamed to avoid conflict
telegramChatId_global = "-4653665640"

def convert_minute_data_interval(df, to_interval=1):
    if to_interval == 1:
        return df
    
    if df is None or df.empty:
        return pd.DataFrame() # Return empty DataFrame if input is empty

    if not isinstance(to_interval, int) or to_interval <= 0:
        raise ValueError("to_interval must be a positive integer.")

    # Ensure 'timestamp' column exists and is in datetime format
    # Assuming the datetime column is named 'timestamp' as per requirements.
    # If it's 'date' from getHistoricalData, it should be handled/renamed before this function
    # or this function should be adapted. For now, proceeding with 'timestamp'.
    if 'timestamp' not in df.columns:
        # Try to use 'date' column if 'timestamp' is missing, assuming it's the datetime column
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'}) 
        else:
            raise ValueError("DataFrame must contain a 'timestamp' or 'date' column for resampling.")
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValueError(f"Could not convert 'timestamp' column to datetime: {e}")

    # The logic below assumes 'instrument_token' exists.
    # If it doesn't, we can add a dummy one since we're resampling a single instrument's data.
    df_copy = df.copy()
    added_dummy_token = False
    if 'instrument_token' not in df_copy.columns:
        df_copy['instrument_token'] = 1 # Dummy token for grouping
        added_dummy_token = True


    # Sort by instrument_token and timestamp
    df_copy = df_copy.sort_values(by=['instrument_token', 'timestamp'])

    # Define base aggregation rules - only use columns that exist
    base_agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Add optional columns if they exist
    optional_columns = {
        'id': 'first',
        'strike': 'first'
    }
    
    # Build final aggregation rules based on available columns
    agg_rules = base_agg_rules.copy()
    for col, agg_func in optional_columns.items():
        if col in df_copy.columns:
            agg_rules[col] = agg_func

    all_resampled_dfs = []

    # Group by instrument_token and then by day for resampling
    # The pd.Grouper will use the 'timestamp' column, group by Day ('D'), using start_day as origin for daily grouping.
    grouped_by_token_day = df_copy.groupby([
        'instrument_token', 
        pd.Grouper(key='timestamp', freq='D', origin='start_day')
    ])

    for (token, day_key), group_data in grouped_by_token_day:
        if group_data.empty:
            continue

        # Define the resampling origin for this specific day: 9:15 AM
        # day_key is the start of the day (00:00:00) from the Grouper
        origin_time_for_day = day_key + pd.Timedelta(hours=9, minutes=15)

        # Set timestamp as index for resampling this group
        group_data_indexed = group_data.set_index('timestamp')
        
        resampled_one_group = group_data_indexed.resample(
            rule=f'{to_interval}min',
            label='left', # Label of the interval is its start time
            origin=origin_time_for_day
        ).agg(agg_rules)

        # Drop rows where 'open' is NaN (meaning no data fell into this resampled interval)
        resampled_one_group = resampled_one_group.dropna(subset=['open'])

        if not resampled_one_group.empty:
            # Add instrument_token back as a column
            resampled_one_group['instrument_token'] = token
            all_resampled_dfs.append(resampled_one_group)
    
    if not all_resampled_dfs:
        return pd.DataFrame(columns=df.columns) # Return empty DF with original columns

    final_df = pd.concat(all_resampled_dfs)
    final_df = final_df.reset_index() # 'timestamp' becomes a column

    if added_dummy_token:
        final_df = final_df.drop(columns=['instrument_token'])

    # Define desired column order
    # (Make sure all these columns exist in final_df after aggregation and reset_index)
    # 'instrument_token' added above, 'timestamp' from reset_index
    base_desired_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    optional_desired_columns = ['ID', 'id', 'strike']
    
    # Build final desired columns list based on what exists
    desired_columns = []
    for col in base_desired_columns:
        if col in final_df.columns:
            desired_columns.append(col)
    
    for col in optional_desired_columns:
        if col in final_df.columns:
            desired_columns.append(col)
    
    # Filter and reorder
    final_df = final_df[[col for col in desired_columns if col in final_df.columns]]
    
    return final_df

class system_initialization:
    def __init__(self):
        
        self.Kite = None
        self.con = None

        # Telegram credentials for instance use
        self.telegram_token = telegramToken_global
        self.telegram_chat_id = telegramChatId_global

        config_file_path = "security.txt"
        with open(config_file_path, 'r') as file:
            content = file.read()

        self.config = ast.literal_eval(content)
        self.api_key = self.config["api_key"]
        self.api_secret = self.config["api_secret"]
        self.userId = self.config["userID"]
        self.pwd = self.config["pwd"]
        self.totp_key = self.config["totp_key"]
        
        self.mysql_username = self.config["username"]
        self.mysql_password = self.config["password"]
        self.mysql_hostname = self.config["hostname"]
        self.mysql_port = int(self.config["port"])
        self.mysql_database_name = self.config["database_name"]
        self.AccessToken = self.config.get("AccessToken") # Use .get() for safety

        print("read security details")

        self.kite = KiteConnect(api_key=self.api_key, timeout=60)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
    
    

    def hard_refresh_access_token(self):
        temp_token = self.kite_chrome_login_generate_temp_token()
        AccessToken = self.kite.generate_session(temp_token, api_secret= self.api_secret)["access_token"]
        self.kite.set_access_token(AccessToken)
        self.SaveAccessToken(AccessToken)

        # Update the in-memory configuration and write to security.txt
        self.config["AccessToken"] = AccessToken # Update self.config dict
        config_file_path = "security.txt"
        try:
            with open(config_file_path, 'w') as file:
                json.dump(self.config, file, indent=4) # Write the whole updated config
            print("Successfully updated AccessToken in security.txt")
        except Exception as update_err:
            print(f"Error updating AccessToken in security.txt: {update_err}")
        
        AccessToken = self.GetAccessToken()
        self.kite.set_access_token(AccessToken)   
        df_nse = self.download_instruments('NSE')
        df_nse['name'] = df_nse['name'].str.replace("'", "", regex=False)
        df_bse = self.download_instruments('BSE')
        df_bse = df_bse[~df_bse['tradingsymbol'].isin(df_nse['tradingsymbol'])]
        df_bse['name'] = df_bse['name'].str.replace("'", "", regex=False)
        
        df_nfo = self.download_instruments('NFO')
        df=pd.concat([df_nse, df_nfo, df_bse], axis=0)
        print("starting DB save")
        self.save_intruments_to_db(data = df)

        print('Ready to trade')
        return self.kite
        
    def init_trading(self):
        self.kite.set_access_token(self.AccessToken)
        try:
            data = self.kite.historical_data(256265,'2025-05-15','2025-05-15','minute')
        except KiteException as e:
            print(e)
            print("Access token expired, Generating new token")
            temp_token = self.kite_chrome_login_generate_temp_token()
            AccessToken = self.kite.generate_session(temp_token, api_secret= self.api_secret)["access_token"]
            self.kite.set_access_token(AccessToken)
            self.SaveAccessToken(AccessToken)

            # Update the in-memory configuration and write to security.txt
            self.config["AccessToken"] = AccessToken # Update self.config dict
            config_file_path = "security.txt"
            try:
                with open(config_file_path, 'w') as file:
                    json.dump(self.config, file, indent=4) # Write the whole updated config
                print("Successfully updated AccessToken in security.txt")
            except Exception as update_err:
                print(f"Error updating AccessToken in security.txt: {update_err}")
            
            AccessToken = self.GetAccessToken()
            self.kite.set_access_token(AccessToken)        

        df_nse = self.download_instruments('NSE')
        df_nse['name'] = df_nse['name'].str.replace("'", "", regex=False)
        df_bse = self.download_instruments('BSE')
        df_bse = df_bse[~df_bse['tradingsymbol'].isin(df_nse['tradingsymbol'])]
        df_bse['name'] = df_bse['name'].str.replace("'", "", regex=False)

        df_nfo = self.download_instruments('NFO')
        # print(df_bse.head())
        # print(df_nse.head())
        df=pd.concat([df_nse, df_nfo, df_bse], axis=0)
        print("starting DB save")
        self.save_intruments_to_db(data = df)
        
        print('Ready to trade')
        # Send initial ready message using the new instance method if OrderPlacement has it,
        # or keep it here if system_initialization itself should send it.
        # For now, let's assume a specific ready message might be sent by LiveTrader itself.
        # Original telegram message send:
        # telegramMessage = 'Ready to trade'
        # telegramURL = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(self.telegram_token,self.telegram_chat_id,telegramMessage)
        # response = requests.get(telegramURL)
        return self.kite

    def GetAccessToken(self):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        cursor = self.con.cursor()
        query = "Select token from daily_token_log where date(created_at) = '{}' order by created_at desc limit 1;".format(datetime.date.today())
        cursor.execute(query)
        print("read token from DB")
        for row in cursor:
            if row is None:
                return ''
            else:
                return row[0]
        self.con.close()
        cursor.close()

    def GetPnL(self):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        query = "Select * from kiteconnect.trades_pnl"
        Pnl = pd.read_sql(query, self.con)
        print("Got PnL")
        self.con.close()
        return Pnl

    def run_query_full(self, query):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        Pnl = pd.read_sql(query, self.con)
        print("Ran Full QUery")
        self.con.close()
        return Pnl

    def run_query_limit(self, query):
        con = None
        cursor = None
        try:
            con = sqlConnector.connect(
                host=self.mysql_hostname, 
                user=self.mysql_username, 
                password=self.mysql_password, 
                database=self.mysql_database_name, 
                port=self.mysql_port,
                auth_plugin='mysql_native_password'
            )
            cursor = con.cursor()
            cursor.execute(query)
            print("run query")
            results_list = cursor.fetchall()
            flat_list = [item[0] for item in results_list]
            return flat_list
        except Exception as e:
            print(f"Database query failed: {e}")
            # Return an empty list or re-raise the exception depending on desired behavior
            return []
        finally:
            # This block is guaranteed to run, ensuring the connection is closed.
            if cursor:
                cursor.close()
            if con and con.is_connected():
                con.close()

    def SaveAccessToken(self,Token):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        q1 = "INSERT INTO kiteConnect.daily_token_log(token) Values('{}')"
        q2 = " ON DUPLICATE KEY UPDATE Token = '{}', created_at=CURRENT_TIMESTAMP();"
        q1 = q1.format(Token)
        q2 = q2.format(Token)
        query = q1 + q2
        cur = self.con.cursor()
        cur.execute(query)
        self.con.commit()
        print("saved token to DB")
        self.con.close()
        cur.close()

    def DeleteData(self,query):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        cur = self.con.cursor()
        cur.execute(query)
        self.con.commit()
        print("Deleted Data")
        self.con.close()
        cur.close()


    def kite_chrome_login_generate_temp_token(self):
        browser = webdriver.Chrome()
        browser.get(self.kite.login_url())
        browser.implicitly_wait(5)

        username = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[1]/input')
        password = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/input') 
        
        username.send_keys(self.userId)
        password.send_keys(self.pwd)
        
        # Click Login Button
        browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[4]/button').click()
        time.sleep(2)

        pin = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div[2]/div/div[2]/form/div[1]/input')
        totp = TOTP(self.totp_key)
        token = totp.now()
        pin.send_keys(token)
        time.sleep(1)
        temp_token=browser.current_url.split('request_token=')[1][:32]
        browser.close()

        print("got temp token")
        
        return temp_token

    def download_instruments(self, exch):
        lst = []
        if exch == 'NSE':
            lst = self.kite.instruments('NSE')
        elif exch == 'NFO':
            lst = self.kite.instruments('NFO') # derivatives NSE
        elif exch == 'BSE':
            lst = self.kite.instruments('BSE')
        elif exch =='CDS':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_CDS) # Currency
        elif exch == 'BFO':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BFO) # Derivatives BSE
        elif exch == 'MCX':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_MCX) # Commodity
        else:
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BCD)
            
        df = pd.DataFrame(lst) # Convert list to dataframe
        if len(df) == 0:
            print('No data returned')
            return
        print("downloading instruments")
        return df

    def save_data_to_db(self, data, tableName):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        print("starting DB save - entered function")
        engine = create_engine("mysql+pymysql://{user}:{pw}@{localhost}:{port}/{db}".format(user=self.mysql_username, localhost = self.mysql_hostname, port = self.mysql_port, pw=self.mysql_password, db=self.mysql_database_name))
        print("starting DB save - created engine")
        data.to_sql(tableName, con = engine, if_exists = 'replace', chunksize = 100000)
        print('Saved to Database')
        self.con.close()

    def save_intruments_to_db(self,data):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        for i in range (0, len(data)):
            query = "insert into kiteConnect.instruments_zerodha values({},'{}','{}','{}',{},'{}',{},{},{},'{}','{}','{}',1) ON DUPLICATE KEY UPDATE instrument_token=instrument_token, is_active=1;".format(data['instrument_token'].iloc[i],data['exchange_token'].iloc[i],data['tradingsymbol'].iloc[i],data['name'].iloc[i],data['last_price'].iloc[i],data['expiry'].iloc[i],data['strike'].iloc[i],data['tick_size'].iloc[i],data['lot_size'].iloc[i],data['instrument_type'].iloc[i],data['segment'].iloc[i],data['exchange'].iloc[i])
            # print(query)
            cur = self.con.cursor()
            cur.execute(query)
            # print("saved token to DB")
        cursor = self.con.cursor()
        cursor.execute("UPDATE kiteconnect.instruments_zerodha a JOIN (Select instrument_token, CASE WHEN sum_volume > 0 THEN 1 ELSE 0 END AS is_active from (SELECT a.instrument_token, sum(b.volume) as sum_volume FROM kiteconnect.instruments_zerodha a LEFT JOIN kiteconnect.historical_data_day b ON a.instrument_token = b.instrument_token AND b.timestamp >= CURDATE() - INTERVAL 15 DAY WHERE a.instrument_type = 'EQ' GROUP BY a.instrument_token) as vol) AS active_flags ON a.instrument_token = active_flags.instrument_token SET a.is_active = active_flags.is_active;")
        self.con.commit()
        print("Instruments active flag updated")
        self.con.close()
        cur.close()

    def close_db_connection(self):
        try:
            self.con.close()
            del self.con
        except:
            pass

class OrderPlacement():
    def __init__(self):
        self.sys_init = system_initialization()

        self.Kite = None
        self.con = None

        # Telegram credentials for instance use
        self.telegram_token = telegramToken_global
        self.telegram_chat_id = telegramChatId_global

        config_file_path = "security.txt"
        with open(config_file_path, 'r') as file:
            content = file.read()

        self.config = ast.literal_eval(content)
        self.api_key = self.config["api_key"]
        self.api_secret = self.config["api_secret"]
        self.userId = self.config["userID"]
        self.pwd = self.config["pwd"]
        self.totp_key = self.config["totp_key"]
        self.AccessToken = self.config.get("AccessToken") # Use .get() for safety
        
        print("read security details")

        self.kite = KiteConnect(api_key=self.api_key, timeout=60)
        
        # self.k_apis = kiteAPIs()  # This seems to cause a circular dependency or redundant initialization
        print("OrderPlacement module initialized. Ensure init_trading() is called if access token needs refresh.")
        
    def init_trading(self):
        self.kite.set_access_token(self.AccessToken)
        try:
            data = self.kite.historical_data(256265,'2025-05-15','2025-05-15','minute')
        except KiteException as e:
            print(e)
            print("Access token expired, Generating new token")
            self.kite = self.sys_init.hard_refresh_access_token()
        print('Ready to trade')

    def send_telegram_message(self, message: str):
        """
        Sends a message to the configured Telegram chat.
        Args:
            message (str): The message text to send.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            print("OrderPlacement: Telegram token or chat ID not configured. Cannot send message.")
            return
        
        telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        params = {
            'chat_id': self.telegram_chat_id,
            'text': message
        }
        try:
            response = requests.get(telegram_url, params=params, timeout=10) # Added timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            print(f"OrderPlacement: Telegram message sent successfully: '{message[:50]}...'")
        except requests.exceptions.RequestException as e:
            print(f"OrderPlacement: Error sending Telegram message: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred sending Telegram message: {e}")

    def place_market_order_live(self, tradingsymbol: str, exchange: str, transaction_type: str, 
                                quantity: int, product: str, tag: str = None):
        """
        Places a market order.
        Args:
            tradingsymbol (str): Trading symbol of the instrument.
            exchange (str): Name of the exchange (e.g., self.kite.EXCHANGE_NFO, self.kite.EXCHANGE_NSE).
            transaction_type (str): Transaction type (self.kite.TRANSACTION_TYPE_BUY or self.kite.TRANSACTION_TYPE_SELL).
            quantity (int): Quantity to transact.
            product (str): Product code (e.g., self.kite.PRODUCT_MIS, self.kite.PRODUCT_NRML).
            tag (str, optional): An optional tag for the order.
        Returns:
            str: Order ID if successful, None otherwise.
        """
        try:
            print(f"[{datetime.datetime.now()}] PLACE_ORDER_LIVE: Preparing to call Kite API.")
            start_time = time.time()
            
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=self.kite.ORDER_TYPE_MARKET,
                tag=tag
            )

            end_time = time.time()
            
            print(f"OrderPlacement: Market order placed for {tradingsymbol}. Order ID: {order_id}")
            return order_id
        except KiteException as e:
            print(f"[{datetime.datetime.now()}] PLACE_ORDER_LIVE: Kite API Exception: {e}")
            # Consider specific error handling or re-raising
        except Exception as e:
            print(f"[{datetime.datetime.now()}] PLACE_ORDER_LIVE: General Exception: {e}")
        return None

    def get_ltp_live(self, instrument_tokens: list[int | str]) -> dict[int, float]:
        """
        Fetches the Last Traded Price (LTP) for a list of instrument tokens.
        This method is optimized for batch requests.

        Args:
            instrument_tokens (list[int | str]): A list of instrument tokens.
                Can be integers or strings.

        Returns:
            dict[int, float]: A dictionary mapping each instrument token (int) to its LTP (float).
                              Returns an empty dictionary if the API call fails or no tokens are provided.
        """
        if not instrument_tokens:
            return {}

        # The Kite Connect API expects a list of instrument tokens.
        try:
            # Ensure the kite session is initialized
            if not self.kite:
                print("OrderPlacement: Kite object not initialized for LTP fetch.")
                return {}

            # ltp() can handle a list of integer or string tokens directly.
            ltp_data = self.kite.ltp(instrument_tokens)
            
            if not ltp_data:
                print(f"OrderPlacement: No LTP data returned for tokens: {instrument_tokens}")
                return {}

            results = {}
            for token_key, data in ltp_data.items():
                if isinstance(data, dict) and 'instrument_token' in data and 'last_price' in data:
                    token_int = int(data['instrument_token'])
                    ltp_value = float(data['last_price'])
                    results[token_int] = ltp_value
                else:
                    # This handles cases where a specific token in the request might have failed.
                    print(f"OrderPlacement: Could not parse LTP data for item: {token_key}={data}")

            # Check if we received LTP for all requested tokens.
            requested_set = set(int(t) for t in instrument_tokens)
            if len(results) != len(requested_set):
                 print(f"OrderPlacement: WARNING: Did not receive LTP for all tokens. Missing: {requested_set - set(results.keys())}")

            return results

        except KiteException as e:
            print(f"OrderPlacement: Kite API error fetching LTP for tokens {instrument_tokens}: {e}")
        except Exception as e:
            print(f"OrderPlacement: General error fetching LTP for tokens {instrument_tokens}: {e}")
        
        return {}

class kiteAPIs:
    def __init__(self):
        self.Kite = None
        self.con = None
        self.startKiteSession = system_initialization()
        self.kite = self.startKiteSession.kite
        self.con = self.startKiteSession.con
        self.api_key = self.startKiteSession.api_key
        self.AccessToken = self.startKiteSession.GetAccessToken()
        self.kite.set_access_token(self.AccessToken)
        self.ticker = KiteTicker(self.startKiteSession.api_key, self.AccessToken)

        self.mysql_username = self.startKiteSession.mysql_username
        self.mysql_password = self.startKiteSession.mysql_password
        self.mysql_hostname = self.startKiteSession.mysql_hostname
        self.mysql_port = self.startKiteSession.mysql_port
        self.mysql_database_name = self.startKiteSession.mysql_database_name



    def extract_holdings_data(self):
        from datetime import datetime, timedelta
        """Extract required fields from holdings JSON data."""
        holdings_data = []
        holdings_json = self.kite.holdings()
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
                'exchange': holding.get('exchange'),
                'average_price': holding.get('average_price'),
                'last_price': holding.get('last_price'),
                'pnl': holding.get('pnl'),
                'pnl_percent': 100.0*holding.get('pnl')/(holding.get('average_price')*holding.get('quantity')),
                'data_source': 'holdings'
            })
        return holdings_data

    def extract_positions_data(self):
        from datetime import datetime, timedelta
        """Extract required fields from positions JSON data (net positions)."""
        positions_data = []
        positions_json = self.kite.positions()
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
                'exchange': position.get('exchange'),
                'average_price': position.get('average_price'),
                'last_price': position.get('last_price'),
                'pnl': position.get('pnl'),
                'pnl_percent': 0 if position.get('quantity') == 0 else 100.0*position.get('pnl')/(position.get('average_price')*position.get('quantity')),
                'data_source': 'positions'
            })
        return positions_data

    # getting the instrument token for a given symbol
    def get_instrument_token(self,symbol):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where tradingsymbol in ({symbol}) and instrument_type = 'EQ'"
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()
    # getting all the instrument tokens for a given instrument type
    def get_instrument_all_tokens(self, instrument_type):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where instrument_type = '{instrument_type}' and is_active = 1".format(instrument_type)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()
    
    def get_instrument_active_tokens(self, instrument_type, end_dt_backfill):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where expiry >= '{end_dt_backfill}' and instrument_type = '{instrument_type}' and is_active = 1".format(instrument_type, end_dt_backfill)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()

    def extract_data_from_db(self, from_date, to_date, interval, instrument_token):
        query = f"SELECT a.*, b.strike FROM kiteConnect.historical_data_{interval} a left join kiteConnect.instruments_zerodha b on a.instrument_token = b.instrument_token where date(a.timestamp) between '{from_date}' and '{to_date}' and a.instrument_token in ({instrument_token})"
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df
    
    def get_trades(self):
        trades = self.kite.trades()
        trades_df = pd.DataFrame(trades)

        trades_df['fill_timestamp'] = trades_df['fill_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_df['order_timestamp'] = trades_df['order_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_df['exchange_timestamp'] = trades_df['exchange_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            data_to_insert = list(trades_df[['trade_id', 'order_id', 'exchange', 'tradingsymbol', 'instrument_token','product', 'average_price', 'quantity', 'exchange_order_id','transaction_type', 'fill_timestamp', 'order_timestamp','exchange_timestamp']].itertuples(index=False, name=None))
        except KeyError as ke:
            print(f"DataFrame missing expected columns for DB insertion: {ke}. Columns available: {df.columns.tolist()}")
            data_to_insert = []

        

        if trades_df is None:
            print("No Trades found")
        else:
            cur = None # Initialize cur to None
            try:
                cur = self.con.cursor()
                query = f"INSERT IGNORE INTO kiteconnect.trades (trade_id, order_id, exchange, tradingsymbol, instrument_token, product, average_price, quantity, exchange_order_id, transaction_type, fill_timestamp, order_timestamp, exchange_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cur.executemany(query, data_to_insert)
                self.con.commit()
            except Exception as e: # Catch potential errors during DB operation
                print(f"Error during bulk insert into kiteconnect.trades: {e}")
            finally:
                if cur:
                    cur.close() # Close cursor
                    self.con.close() # Close connection after each token's DB operations
    
    def getHistoricalData(self, from_date, to_date, tokens, interval):
        # embed()
        if from_date > to_date:
            return
        
        if tokens == -1:
            print('Invalid symbol provided')
            return 'None'
        
        i = 0

        token_exceptions = []
        MAX_RETRIES = 3
        RETRY_DELAY_SECONDS = 5

        for t in tokens:
            # print(f"Fetching data for token: {t}")
            records = None # Initialize records to None
            for attempt in range(MAX_RETRIES):
                try:
                    # print(f"  Attempt {attempt + 1}/{MAX_RETRIES} for token {t}...")
                    records = self.kite.historical_data(t, from_date=from_date, to_date=to_date, interval=interval)
                    print(f"  Successfully fetched data for token {t} on attempt {attempt + 1}.")
                    break  # Success, exit retry loop
                except (KiteException, requests.exceptions.ReadTimeout, requests.exceptions.RequestException) as e:
                    print(f"  Error on attempt {attempt + 1} for token {t}: {type(e).__name__} - {str(e)}")
                    if str(e) == 'invalid token':
                        print(f"  Invalid token {t} found. Skipping.")
                        break
                    elif attempt < MAX_RETRIES - 1 and str(e) != 'invalid token':
                        print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        self.startKiteSession.hard_refresh_access_token()
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(f"  Max retries reached for token {t}. Adding to exceptions.")
                        token_exceptions.append(t)
                        records = [] # Ensure records is an empty list on failure to avoid issues later
                except Exception as e: # Catch any other unexpected errors
                    print(f"  An unexpected error occurred on attempt {attempt + 1} for token {t}: {type(e).__name__} - {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        self.startKiteSession.hard_refresh_access_token()
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(f"  Max retries reached for token {t} due to unexpected error. Adding to exceptions.")
                        token_exceptions.append(t)
                        records = [] # Ensure records is an empty list
                        # Optionally re-raise the last exception if it's critical and unhandled by appending to token_exceptions
                        # raise
            
            # Continue with processing if records were fetched
            if records is not None and len(records) > 0:
                df = pd.DataFrame(records) # Convert to DataFrame here
                df['instrument_token'] = t
                # df = pd.concat([df, records_df], axis = 0) # This concat was for an older structure
                df['interval'] = interval
                df['id'] = df['instrument_token'].astype(str) + '_' + df['interval'] + '_' + pd.to_datetime(df['date']).dt.strftime("%Y%m%d%H%M") # Ensure date is used correctly
            
                self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
# Ensure 'date' column is pandas Timestamp for intermediate operations if not already
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])

                # Convert pandas Timestamp to string in 'YYYY-MM-DD HH:MM:SS' format for DB insertion
                # This matches how it would likely be formatted implicitly in row-by-row string insertion
                df['date_for_db'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # Prepare data for executemany: list of tuples
                # Column order matches the VALUES clause: id, instrument_token, date_for_db (string for timestamp col), open, high, low, close, volume
                try:
                    data_to_insert = list(df[['id', 'instrument_token', 'date_for_db', 'open', 'high', 'low', 'close', 'volume']].itertuples(index=False, name=None))
                except KeyError as ke:
                    print(f"DataFrame missing expected columns for DB insertion: {ke}. Columns available: {df.columns.tolist()}")
                    data_to_insert = [] 

                target_table_name = None
                if interval == 'minute':
                    target_table_name = "kiteConnect.historical_data_minute"
                elif interval == 'day':
                    target_table_name = "kiteConnect.historical_data_day"
                

                if target_table_name and data_to_insert:
                    cur = None # Initialize cur to None
                    try:
                        cur = self.con.cursor()
                        # Using %s placeholders for values in the query
                        query = f"INSERT IGNORE INTO {target_table_name} (id, instrument_token, timestamp, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                        # Assuming your table columns are named: id, instrument_token, timestamp (for date), open, high, low, close, volume
                        # If your table does not explicitly name columns in this order, adjust the query or ensure table was created with this order.
                        # For safety, it's best if INSERT specifies column names, like above.
                        # If you must stick to the original schemaless insert:
                        # query = f"INSERT IGNORE INTO {target_table_name} VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"


                        # print(f"Attempting to bulk insert {len(data_to_insert)} records into {target_table_name}...")
                        cur.executemany(query, data_to_insert)
                        self.con.commit()
                        # print(f"Bulk insert/ignore completed for {target_table_name}. Rows affected: {cur.rowcount}")
                    except Exception as e: # Catch potential errors during DB operation
                        print(f"Error during bulk insert into {target_table_name}: {e}")
                        # Consider self.con.rollback() if an error occurs and transactions are being used explicitly
                    finally:
                        if cur:
                            cur.close() # Close cursor
                            self.con.close() # Close connection after each token's DB operations
                elif not data_to_insert:
                    print(f"No data prepared for insertion for interval {interval} (data_to_insert list is empty).")
                else:
                    print(f"Unsupported interval '{interval}' or no table determined for database insertion.")
            elif records is not None and len(records) == 0: # Successfully fetched, but no data for the period
                 print(f"No historical data returned for token {t} for the given period.")
            # If records is None (all retries failed), it's already added to token_exceptions.
            
       
        print('token_exceptions: ',token_exceptions)
        # The original code returned 'df' which was the DataFrame for the *last* successfully processed token.
        # This might not be the desired behavior if processing multiple tokens.
        # If the goal is to collect all data, it should be aggregated.
        # For now, I'll keep it returning the last df, but this is a point of attention.
        # If all tokens fail, df might be uninitialized or from a much earlier success.
        # A safer return would be a list of dataframes or a concatenated one if all are successful.
        # Given the current structure, if all fail, df from previous loop iteration might be returned.
        # It's better to initialize df to an empty DataFrame at the start of the outer loop or handle returns more carefully.
        
        # Let's adjust to return an aggregated DataFrame or an empty one if all fail.
        all_data_dfs = []
        # The processing logic (DB insertion, etc.) should be inside the token loop if df is defined per token.
        # The current edit places the retry loop inside the token loop, and df is created from 'records'.
        # If the intention is to return a single DataFrame containing all data, then aggregation is needed.
        # The provided edit does not aggregate `df` across tokens. It processes one token at a time.
        # The `return df` at the end will return the DataFrame of the LAST token processed (or an error if it failed).
        # This seems consistent with the original structure where `df = pd.concat([df, records], axis = 0)` was commented out.
        # If you want to return *all* data fetched, we'd need to collect each token's df into a list and concat at the end.

        # For now, to ensure df is defined even if the last token fails but previous ones succeeded,
        # we should initialize df outside the loop, or adjust the return.
        # The current logic processes and potentially saves data per token. The final 'df' return is for the last token.
        # If the last token processing fails and `records` becomes `[]`, then `df` will be an empty DataFrame for that token.
        
        # The `df` variable is defined *inside* the `if records is not None and len(records) > 0:` block.
        # If the last token fails all retries, `records` will be `[]`, this block will be skipped,
        # and `df` might not be defined for the return if it was the *only* token.
        # Let's ensure df is initialized if we intend to always return it.
        
        # Given the original context, it seems `df` was intended to be the data for the current token being processed.
        # The return `df` at the very end is problematic if there are multiple tokens.
        # The primary action is saving to DB.
        # If a return value is truly needed for all data, a list of DFs or a concatenated DF should be built.
        # For now, I'll stick to the modified logic where `df` is per-token for DB saving and the final `return df`
        # will be the last token's data (or empty if it failed).
        # The `token_exceptions` list is the primary indicator of failures.

        # A small correction: Ensure 'df' is defined if the loop completes.
        # However, the current logic uses `df` within the loop for DB operations.
        # The final `return df` is likely a remnant or for a single-token use case.
        # If this function is always called with multiple tokens and an aggregated result is expected,
        # this return value needs rethinking. For now, I'll assume the DB saving is the main goal per token.

        # The simplest fix for the return value, if it must return *something* even if all fail,
        # is to initialize an empty df at the beginning of the function.
        # However, the loop structure processes one token at a time.
        
        # Let's assume the function's primary purpose is to fetch and store,
        # and the return value might be for convenience for the last token.
        # If 'records' is empty after retries, df won't be formed for that token.
        # The main thing is that `token_exceptions` tracks failures.
        # The print statements indicate progress.
        # The changes below focus on the retry logic as requested.
        final_df_to_return = pd.DataFrame() # Default empty return
        if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty: # If df was defined and has data from the last token
            final_df_to_return = df
        
        return final_df_to_return # Return df of the last processed token or empty if all failed / last one failed

    
