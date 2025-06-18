import pandas as pd
import argparse
import os

def convert_parquet_to_csv(file_path):
    """
    Reads a Parquet file and saves it as a CSV file in the same directory.
    """
    # Validate input file path
    if not file_path.endswith('.parquet'):
        print("Error: Input file must be a .parquet file.")
        return
        
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Reading Parquet file: {file_path} ---")
    df = pd.read_parquet(file_path)
    
    
    
    
    # Define output path
    output_path = file_path.replace('.parquet', '.csv')
    
    print(f"--- Saving as CSV to: {output_path} ---")
    # df.to_csv(output_path)
    
    total_signals = df[df['signal'] == 1].shape[0]
    return total_signals
    print("--- Conversion Complete ---")


if __name__ == '__main__':
    signals_dir = 'data/signals'
    summary_data = []
    for filename in sorted(os.listdir(signals_dir)):
        if not filename.endswith('_signals.parquet'):
            continue
        file_path = os.path.join(signals_dir, filename)
        
        # Assuming convert_parquet_to_csv returns a DataFrame
        signals = convert_parquet_to_csv(file_path)
        summary_data.append({'filename': filename, 'signal_count': signals})
    
    if summary_data:
        signal_summary = pd.DataFrame(summary_data)
        output_path = os.path.join(signals_dir, 'signal_summary.csv')
        signal_summary.to_csv(output_path, index=False)
        print(f"Signal summary saved to {output_path}")
    else:
        print("No signal files found to summarize.")