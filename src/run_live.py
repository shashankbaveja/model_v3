import os
import sys
import subprocess
from datetime import datetime
import yaml
import pandas as pd

# --- Configuration ---
LOG_DIR = 'logs'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myKiteLib import OrderPlacement

def run_step(command, log_file):
    """Executes a command as a subprocess and logs its output."""
    print(f"\n{'='*25}\nRUNNING: {' '.join(command)}\n{'='*25}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output to console and log file in real-time
        with open(log_file, 'a') as f:
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
        
        process.wait() # Wait for the subprocess to finish
        
        if process.returncode != 0:
            error_msg = f"!!! Step failed with return code {process.returncode} !!!"
            print(error_msg)
            with open(log_file, 'a') as f:
                f.write(f"\n{error_msg}\n")
            # Decide if you want to stop the pipeline on failure
            # sys.exit(1) 
            return False # Indicate failure
            
    except Exception as e:
        error_msg = f"!!! An exception occurred while running the step: {e} !!!"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(f"\n{error_msg}\n")
        # sys.exit(1)
        return False # Indicate failure
        
    return True # Indicate success


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

def main():
    """Main pipeline execution function."""
    
    # --- Setup ---
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"live_run_{timestamp}.log")
    
    python_executable = '/opt/anaconda3/envs/KiteConnect/bin/python'
    
    print(f"--- Starting Full Pipeline Run ---")
    print(f"Logging output to: {log_file}")
    
    # --- Pipeline Steps ---
    
    # Step 1: Feature Generation (Base features and targets)
    # This script will need to be modified to accept config for the new target
    
    if not run_step([python_executable,  '-u', 'src/data_backfill/data_backfill_daily.py'], log_file):
        print("Stopping pipeline due to failure in data backfill.")
        sys.exit(1)


    if not run_step([python_executable,  '-u', 'src/data_pipeline.py'], log_file):
        print("Stopping pipeline due to failure in data pipeline.")
        sys.exit(1)

    if not run_step([python_executable,  '-u', 'src/feature_generator.py'], log_file):
        print("Stopping pipeline due to failure in feature generation.")
        sys.exit(1)

    if not run_step([python_executable,  '-u', 'src/pattern_feature_generator.py'], log_file):
        print("Stopping pipeline due to failure in pattern feature generation.")
        sys.exit(1)
        
    # Step 3: Merge Pattern Features
    if not run_step([python_executable,  '-u', 'src/merge_features.py'], log_file):
        print("Stopping pipeline due to failure in feature merging.")
        sys.exit(1)

    # Step 6: Signal Generation
    if not run_step([python_executable,  '-u', 'src/signal_generator.py'], log_file):
        print("Stopping pipeline due to failure in signal generation.")
        sys.exit(1)

    # Step 7: Ask Gemini for confirmation on trades
    if not run_step([python_executable,  '-u', 'src/gemini_bridge.py'], log_file):
        print("Stopping pipeline due to failure in gemini bridge.")
        sys.exit(1)
    
    print(f"\n--- Pipeline Finished Successfully ---")
    print(f"Full log available at: {log_file}")


if __name__ == "__main__":
    config = load_config()
    order_placement = OrderPlacement()
    order_placement.send_telegram_message(f"Starting Live Run for date {config['data']['test_end_date']}")
    main() 

    
    