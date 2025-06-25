import pandas as pd
import joblib
import os
import sys
from datetime import datetime

# Add the parent directory to the Python path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config

log_file_path = "reports/trades/daily_trades.csv"

def generate_all_signals():
    """
    Loads all trained models, generates trade signals based on specified thresholds,
    and saves them as new datasets containing OHLCV and signal data.
    """
    print("--- Starting Signal Generation Pipeline ---")
    config = load_config()

    # Get parameters from config
    model_config = config.get('modeling', {})
    data_config = config.get('data', {})
    strategies = model_config.get('strategies_to_train', ['momentum', 'reversion', 'combined'])
    model_types = model_config.get('model_types', ['lightgbm'])
    thresholds = config.get('trading', {}).get('backtest_thresholds', [0.6])

    # --- Setup Directories ---
    processed_dir = 'data/processed'
    signals_dir = 'data/signals'
    models_dir = 'models'
    os.makedirs(signals_dir, exist_ok=True)

    # --- Load Raw Price Data ---
    # This contains the OHLCV data we need to join with the signals.
    try:
        raw_test_data = pd.read_parquet(os.path.join(processed_dir, 'test_raw.parquet'))
        print("Loaded raw test data for OHLCV information.")
    except FileNotFoundError as e:
        print(f"Error: Raw test data not found. Details: {e}")
        sys.exit(1)

    # --- Main Loop for Signal Generation ---
    for strategy in strategies:
        print(f"\nProcessing strategy: {strategy.upper()}")

        # Load the corresponding feature data for the strategy
        feature_data_path = os.path.join(processed_dir, f'test_{strategy}_with_patterns_features.parquet')
        try:
            feature_df = pd.read_parquet(feature_data_path)
            print(f"  Loaded feature data from {feature_data_path}")
        except FileNotFoundError:
            print(f"Warning: Feature data not found for strategy '{strategy}'. Skipping.")
            continue
        
        # Prepare feature matrix X_test by dropping non-feature columns
        X_test = feature_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

        for direction in ['up']:
            for model_type in model_types:
                
                model_name = f'{strategy}_{direction}_{model_type}_model.joblib'
                model_path = os.path.join(models_dir, model_name)

                if not os.path.exists(model_path):
                    print(f"  - Model not found: {model_path}. Skipping.")
                    continue

                # Load the model and predict probabilities
                print(f"  - Generating signals for: {model_name}")
                model = joblib.load(model_path)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                for threshold in thresholds:
                    # Create signal column based on the threshold
                    signals = (y_pred_proba >= threshold).astype(int)
                    signals_prob = y_pred_proba

                    # Create a temporary DataFrame with identifiers and the signal
                    temp_df = feature_df[['instrument_token', 'timestamp']].copy()
                    temp_df['signal'] = signals
                    temp_df['signal_prob'] = signals_prob
                    
                    # Merge with raw data to get OHLCV columns
                    # This ensures the final dataset has all necessary info for backtesting
                    final_signal_df = pd.merge(raw_test_data, temp_df, on=['instrument_token', 'timestamp'])

                    # --- Print signals generated for today ---
                    todays_signals = final_signal_df[
                        (pd.to_datetime(final_signal_df['timestamp']).dt.date == pd.to_datetime('today').date()) &
                        (final_signal_df['signal'] == 1)
                    ]
                    if not todays_signals.empty:
                        print(f"    >>> Today's Signals ({model_name}, thresh {threshold:.2f}):")
                        print(todays_signals[['instrument_token', 'close']])
                        print("-" * 40)
                    else:
                        print(f"    >>> No signals generated for today ({model_name}, thresh {threshold:.2f})")
                    # --- End of section ---

                    # Save the final signal file
                    # output_filename = f'{strategy}_{direction}_{model_type}_thresh_{threshold:.2f}_signals.parquet'
                    # output_path = os.path.join(signals_dir, output_filename)
                    # final_signal_df.to_parquet(output_path, index=False)

                    output_filename = f'{strategy}_{direction}_{model_type}_thresh_{threshold:.2f}_signals.csv'
                    output_path = os.path.join(signals_dir, output_filename)
                    final_signal_df.to_csv(output_path, index=False)

                    data_end_date = data_config.get('test_end_date', datetime.now().strftime('%Y-%m-%d'))
                    signal_df_for_gemini = final_signal_df[final_signal_df['timestamp'] == data_end_date][['instrument_token', 'signal', 'signal_prob']]
                    signal_df_for_gemini = signal_df_for_gemini[signal_df_for_gemini['signal'] == 1][['instrument_token', 'signal', 'signal_prob']]
                    
                    signal_df_for_gemini.to_csv(log_file_path, index=False)
                
                    print(f"    - Saved signals to {output_path}")

    print("\n--- Signal Generation Pipeline Finished ---")

if __name__ == "__main__":
    generate_all_signals() 