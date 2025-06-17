import pandas as pd
import os
import sys

# Add the parent directory to the Python path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config

def merge_features():
    """
    This script merges the newly generated harmonic pattern features with the
    existing momentum and reversion feature sets for a specific time interval.
    """
    # --- Configuration ---
    
    PROCESSED_DIR = 'data/features'
    FEATURES_DIR = 'data/features'
    OUTPUT_DIR = 'data/processed' # Save back to the processed dir for the pipeline
    
    # --- Load Pattern Features ---
    # The pattern file is now generated for multiple instruments.
    base_pattern_file = os.path.join(FEATURES_DIR, 'MULTI_INSTRUMENT_pattern_features.parquet')
    pattern_features_path = base_pattern_file
    
    print(f"Loading pattern features from {pattern_features_path}...")
    try:
        pattern_features_df = pd.read_parquet(pattern_features_path)
    except FileNotFoundError:
        print(f"ERROR: Pattern features file not found. Please run pattern_feature_generator.py first.")
        sys.exit(1)

    # We only need the key columns and the new pattern columns for the merge.
    pattern_cols = [col for col in pattern_features_df.columns if col.startswith('pattern_')]
    patterns_only_df = pattern_features_df[['instrument_token', 'timestamp'] + pattern_cols]
    print(f"Found {len(pattern_cols)} pattern feature columns.")

    # --- Define feature sets and partitions to process ---
    config = load_config('config/parameters.yml')
    feature_types = config.get('strategies_to_train', ['momentum', 'reversion','combined'])
    partitions = ['train', 'validation', 'test']

    # --- Loop, Merge, and Save ---
    for f_type in feature_types:
        for part in partitions:
            # Construct the interval-specific input path
            base_file_name = f'{part}_{f_type}_features.parquet'
            input_path = os.path.join(PROCESSED_DIR, base_file_name)
            
            # The 'all' feature set might not exist for train/validation/test, so we check.
            if f_type == 'all':
                base_file_name = f'{part}_all_features.parquet'
                input_path = os.path.join(PROCESSED_DIR, base_file_name)

            if not os.path.exists(input_path):
                print(f"Skipping: Base feature file not found at {input_path}")
                continue

            print(f"Processing: {input_path}")

            # Load the base feature set (e.g., train_momentum_features.parquet)
            base_df = pd.read_parquet(input_path)

            # Join the pattern features on both instrument and timestamp.
            merged_df = pd.merge(
                base_df,
                patterns_only_df,
                on=['instrument_token', 'timestamp'],
                how='left'
            )

            # After merging, pattern columns will be NaN for rows that didn't have a pattern.
            # We fill these with 0.
            merged_df[pattern_cols] = merged_df[pattern_cols].fillna(0).astype(int)

            # Define the new interval-specific output filename, using 'with_patterns' suffix
            # to indicate enrichment.
            output_file_name = f'{part}_{f_type}_with_patterns_features.parquet'
            output_path = os.path.join(OUTPUT_DIR, output_file_name)

            print(f"Saving enriched feature set to {output_path}...")
            # Save as a flat file, consistent with other processed datasets.
            merged_df.to_parquet(output_path, index=False)

    print("\nFeature merging process completed successfully.")

if __name__ == '__main__':
    merge_features() 