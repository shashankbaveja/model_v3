import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
import talib

# Add the parent directory to the Python path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# List of all candlestick patterns from TA-Lib to be generated
# As requested by the user.
ALL_TA_PATTERNS = [
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
    'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
    'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
    'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
    'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
    'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
    'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
    'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
    'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
    'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN',
    'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
    'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
    'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
]

def generate_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates ternary features for all TA-Lib candlestick patterns.
    This function now supports multi-instrument dataframes by grouping
    operations by the 'instrument_token' column.

    This function iterates through a predefined list of TA-Lib pattern
    recognition functions. For each pattern, it creates a single column
    with values: 1 for a bullish signal, -1 for a bearish signal, and 0
    for no pattern.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLC and 'instrument_token' columns.
                           It must contain 'open', 'high', 'low', 'close' columns.
                           The index should be the timestamp.

    Returns:
        pd.DataFrame: The original DataFrame with added columns for each
                      detected candlestick pattern signal.
    """
    print("Starting TA-Lib candlestick pattern feature generation...")
    
    # Helper function to apply TA-Lib logic to a single instrument's data
    def _apply_patterns_to_group(group):
        # This dataframe will hold the patterns for the current group
        patterns_df = pd.DataFrame(index=group.index)
        
        # Ensure columns are float for TA-Lib
        ohlc = {
            'open': group['open'].astype(float),
            'high': group['high'].astype(float),
            'low': group['low'].astype(float),
            'close': group['close'].astype(float)
        }

        for pattern_name in ALL_TA_PATTERNS:
            try:
                # Dynamically call the TA-Lib function for the pattern
                pattern_function = getattr(talib, pattern_name)
                result = pattern_function(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])

                # TA-Lib returns 100 for bullish, -100 for bearish, 0 for no pattern.
                if (result != 0).any():
                    patterns_df[f'pattern_{pattern_name}'] = (result / 100).astype(int)

            except Exception as e:
                # Silently fail for any given pattern if it errors out
                pass
                
        return patterns_df

    # Set timestamp as index if it's not already
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp', drop=False)
        
    # tqdm setup for pandas apply
    tqdm.pandas(desc="Generating Candlestick Pattern Features")
    
    # Group by instrument and apply pattern generation
    # The result will have a multi-index (instrument_token, timestamp)
    all_patterns_df = df.groupby('instrument_token').progress_apply(_apply_patterns_to_group)
    
    print(f"\nGenerated {len(all_patterns_df.columns)} new candlestick pattern features.")
    
    # --- Correctly merge the features back to the original data ---
    # To ensure a correct merge, we reset the index on both dataframes
    # so we can merge on the columns 'instrument_token' and 'timestamp'.
    df_reset = df.reset_index()
    patterns_reset = all_patterns_df.reset_index()
    
    output_df = pd.merge(df_reset, patterns_reset, on=['instrument_token', 'timestamp'], how='left')
    
    # The merge might create NaNs for timestamps that don't align; fill with 0
    pattern_cols = [col for col in output_df.columns if col.startswith('pattern_')]
    output_df[pattern_cols] = output_df[pattern_cols].fillna(0).astype(int)
    
    # Restore the timestamp index for consistency with the function's expected output
    output_df.set_index('timestamp', inplace=True)
    
    print("TA-Lib pattern feature generation complete.")
    return output_df

if __name__ == '__main__':
    # This is an example of how to use the generator.
    # It loads the raw partitioned data, combines it, runs the generator,
    # and saves the result as a single feature file.
    
    print("Running pattern feature generator in standalone mode...")
    
    DATA_DIR = 'data/processed'
    OUTPUT_DIR = 'data/features'
    # The output is now for multiple instruments, not just one.
    INSTRUMENT_NAME = 'MULTI_INSTRUMENT'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load the partitioned raw data files
    try:
        print("Loading and concatenating raw 1-minute data files...")
        train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_raw.parquet'))
        validation_df = pd.read_parquet(os.path.join(DATA_DIR, 'validation_raw.parquet'))
        test_df = pd.read_parquet(os.path.join(DATA_DIR, 'test_raw.parquet'))
        
        # Combine and sort the data to form a continuous timeline
        ohlc_df = pd.concat([train_df, validation_df, test_df]).sort_values('timestamp')
        
        # Ensure the index is a DatetimeIndex and instrument_token is a column
        ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp'])
        ohlc_df.set_index('timestamp', inplace=True)

    except FileNotFoundError:
        print(f"ERROR: Raw data files not found in {DATA_DIR}")
        print("Please ensure you have run the initial data processing script.")
        sys.exit(1)
        
    print(f"Loaded a total of {len(ohlc_df)} 1-minute candles for {INSTRUMENT_NAME}.")

    # Generate the pattern features
    df_with_patterns = generate_pattern_features(ohlc_df)
    
    # Save the results
    base_output_filename = f'{INSTRUMENT_NAME}_pattern_features.parquet'
    output_filepath = os.path.join(OUTPUT_DIR, base_output_filename)
    
    print(f"Saving features to {output_filepath}...")
    # The dataframe index is the timestamp, reset it to save as a flat column.
    df_with_patterns.reset_index(inplace=True)
    df_with_patterns.to_parquet(output_filepath, index=False)
    
    print("\n--- Summary ---")
    pattern_columns = [col for col in df_with_patterns.columns if col.startswith('pattern_')]
    if pattern_columns:
        print("Detected pattern counts (total occurrences):")
        # Calculate total occurrences (bullish + bearish) and display
        patterns_summary_df = df_with_patterns[pattern_columns]
        total_counts = (patterns_summary_df != 0).sum().sort_values(ascending=False)
        print(total_counts[total_counts > 0])
    else:
        print("No patterns were detected in the provided data.")
    print("-----------------\n")

    print("Script finished successfully.") 