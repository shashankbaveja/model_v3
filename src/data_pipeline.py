import yaml
import pandas as pd
import os
import sys
from datetime import datetime

# Add the root directory to the Python path to allow importing 'myKiteLib'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import kiteAPIs, system_initialization

def load_config(config_path='config/parameters.yml'):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def fetch_and_clean_data(api_client, sys_details, config):
    """Fetches raw data from the database and performs initial cleaning."""
    print("Fetching data from the database...")
    data_config = config['data']
    tokenList = sys_details.run_query_limit(data_config['token_list_query'])
    tokenList = map(str, tokenList)
    tokenList = ', '.join(tokenList)
    df = api_client.extract_data_from_db(
        from_date=data_config['training_start_date'],
        to_date=data_config['test_end_date'],
        interval='day',
        instrument_token=tokenList
    )
    
    if df is None or df.empty:
        print("Error: No data returned from the database. Exiting.")
        sys.exit(1)
        
    print(f"Successfully fetched {len(df)} rows.")
    
    # --- Data Cleaning ---
    print("Cleaning data...")
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by time
    df.sort_values(by='timestamp', inplace=True)
    
    # Drop any duplicate timestamps
    df.drop_duplicates(subset=['timestamp','instrument_token'], keep='first', inplace=True)
    
    print("Data cleaning complete.")
    print(len(df))
    return df

def split_data(df, config):
    """Splits the dataframe into training, validation, and test sets."""
    print("Splitting data into training, validation, and test sets...")
    data_config = config['data']
    
    # Convert config dates to datetime for comparison
    train_start = pd.to_datetime(data_config['training_start_date'])
    train_end = pd.to_datetime(data_config['training_end_date']).replace(hour=23, minute=59, second=59)
    
    validation_start = train_end + pd.Timedelta(seconds=1)
    validation_end = pd.to_datetime(data_config['validation_end_date']).replace(hour=23, minute=59, second=59)
    
    test_start = validation_end + pd.Timedelta(seconds=1)
    test_end = pd.to_datetime(data_config['test_end_date']).replace(hour=23, minute=59, second=59)
    
    # Create a date mask for splitting
    train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    validation_df = df[(df['timestamp'] >= validation_start) & (df['timestamp'] <= validation_end)]
    test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)]
    
    print(f"Training set: {len(train_df)} rows")
    print(f"Validation set: {len(validation_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    return train_df, validation_df, test_df

def save_datasets(train_df, validation_df, test_df, output_dir='data/processed'):
    """Saves the datasets to the specified directory."""
    print(f"Saving datasets to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(output_dir, 'train_raw.parquet'))
    validation_df.to_parquet(os.path.join(output_dir, 'validation_raw.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test_raw.parquet'))
    
    print("Datasets saved successfully.")

def main():
    """Main function to run the data ingestion pipeline."""
    print("--- Starting Data Ingestion Pipeline ---")
    
    config = load_config()
    # if config['data']['test_end_date'] != datetime.now().strftime('%Y-%m-%d'):
    #     print("Today's Date is not updated, do you want to continue? (y/n)")
        
    #     while True:
    #         user_input = input().strip().lower()
    #         if user_input == 'y':
    #             print("Continuing with the existing test_end_date...")
    #             break
    #         elif user_input == 'n':
    #             print("Exiting script as requested.")
    #             sys.exit(0)
    #         else:
    #             print("Please enter 'y' to continue or 'n' to exit:")
    
    api_client = kiteAPIs()
    sys_details = system_initialization()
    
    full_df = fetch_and_clean_data(api_client, sys_details, config)
    
    train_df, validation_df, test_df = split_data(full_df, config)
    
    save_datasets(train_df, validation_df, test_df)
    
    print("--- Data Ingestion Pipeline Finished ---")

if __name__ == "__main__":
    main()
