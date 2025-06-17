import os
import sys
import subprocess
from datetime import datetime
import yaml
import pandas as pd

# --- Configuration ---
LOG_DIR = 'logs'

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

def generate_summary_report(log_file):
    """Loads the intermediate reports and generates a final summary."""
    print(f"\n{'='*25}\nGenerating Final Summary Report\n{'='*25}")
    
    try:
        metrics_df = pd.read_csv('reports/classification_metrics.csv')
        pnl_df = pd.read_csv('reports/pnl_summary.csv')
    except FileNotFoundError as e:
        msg = f"Could not generate summary report. Missing result file: {e}"
        print(msg)
        with open(log_file, 'a') as f:
            f.write(f"\n{msg}\n")
        return

    # Ensure Threshold columns are of the same type for merging
    metrics_df['Threshold'] = metrics_df['Threshold'].astype(float)
    pnl_df['Threshold'] = pnl_df['Threshold'].astype(float)

    # Merge the two dataframes
    summary_df = pd.merge(metrics_df, pnl_df, on=['Model Name', 'Threshold'], how='left')

    # Reorder and format columns for readability
    summary_df = summary_df[[
        'Strategy', 'Direction', 'Algorithm', 'Threshold',
        'Precision', 'Recall', 'F1-Score', 'Signals Predicted', 'Total Trades',
        'Total Return Pct', 'Win Rate Pct', 'Profit Factor', 'Max Drawdown Pct', 'Sharpe Ratio'
    ]]

    # Sort for clarity
    summary_df.sort_values(by=['Strategy', 'Total Return Pct'], ascending=[True, False], inplace=True)
    
    report_string = summary_df.to_string()
    
    print(report_string)
    with open(log_file, 'a') as f:
        f.write("\n\n--- FINAL CONSOLIDATED REPORT ---\n")
        f.write(report_string)

def main():
    """Main pipeline execution function."""
    
    # --- Setup ---
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"pipeline_run_{timestamp}.log")
    
    python_executable = '/opt/anaconda3/envs/KiteConnect/bin/python'
    
    print(f"--- Starting Full Pipeline Run ---")
    print(f"Logging output to: {log_file}")
    
    # --- Pipeline Steps ---
    
    # Step 1: Feature Generation (Base features and targets)
    # This script will need to be modified to accept config for the new target
    
    if not run_step([python_executable, 'src/data_pipeline.py'], log_file):
        print("Stopping pipeline due to failure in data pipeline.")
        sys.exit(1)

    if not run_step([python_executable, 'src/feature_generator.py'], log_file):
        print("Stopping pipeline due to failure in feature generation.")
        sys.exit(1)

    if not run_step([python_executable, 'src/pattern_feature_generator.py'], log_file):
        print("Stopping pipeline due to failure in pattern feature generation.")
        sys.exit(1)
        
    # Step 3: Merge Pattern Features
    if not run_step([python_executable, 'src/merge_features.py'], log_file):
        print("Stopping pipeline due to failure in feature merging.")
        sys.exit(1)

    # Step 4: Train All Models
    if not run_step([python_executable, 'src/train_model.py'], log_file):
        print("Stopping pipeline due to failure in model training.")
        sys.exit(1)

    # Step 5: Evaluate All Models (Classification Metrics)
    if not run_step([python_executable, 'src/evaluate_model.py'], log_file):
        print("Stopping pipeline due to failure in model evaluation.")
        sys.exit(1)

    # Step 6: Signal Generation
    if not run_step([python_executable, 'src/signal_generator.py'], log_file):
        print("Stopping pipeline due to failure in signal generation.")
        sys.exit(1)

    # Step 6: Run All Backtests (PnL Evaluation)
    if not run_step([python_executable, 'src/backtests.py'], log_file):
        print("Pipeline finished, but backtesting step failed.")
        # We still want to try and generate a summary if some steps completed
        generate_summary_report(log_file) 
        sys.exit(1)
        
    # --- Final Summary ---
    generate_summary_report(log_file)

    print(f"\n--- Pipeline Finished Successfully ---")
    print(f"Full log available at: {log_file}")


if __name__ == "__main__":
    main() 