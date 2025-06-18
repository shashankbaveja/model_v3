import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
from sklearn.pipeline import Pipeline
# Add the root directory for local library import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_pipeline import load_config

def evaluate_model(model_path, X_test, y_test, threshold=0.5):
    """Evaluates a single trained model."""
    model = joblib.load(model_path)
    
    # Predict probabilities and apply threshold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Signals Predicted': y_pred.sum()
    }

def main():
    """
    Main function to evaluate all trained models based on the current config
    and generate a consolidated report.
    """
    print("--- Starting Model Evaluation ---")
    config = load_config('config/parameters.yml')
    # Get parameters from the unified config
    model_config = config.get('modeling', {})
    strategies = model_config.get('strategies_to_train', ['momentum', 'reversion','combined'])
    model_types = model_config.get('model_types', ['lightgbm'])
    processed_dir = 'data/processed'
    # Evaluate at all the thresholds we will backtest
    backtest_thresholds = config['trading']['backtest_thresholds']

    # --- Load All Test Data ---
    print(f"Loading test datasets")
    test_data = {}
    for s in strategies:
        try:
            base_path = os.path.join(processed_dir, f'test_{s}_with_patterns_features.parquet')
            path = base_path
            test_data[s] = pd.read_parquet(path)
            
        except FileNotFoundError:
            print(f"Warning: Test data for strategy '{s}' not found at {path}. Skipping.")

    if not test_data:
        print("Error: No test data could be loaded. Exiting.")
        sys.exit(1)

    # --- Evaluate All Models based on config ---
    results = []

    for strategy_name in strategies:
        for direction in ['up', 'down']:
            for model_type in model_types:
                base_model_filename = f'{strategy_name}_{direction}_{model_type}_model.joblib'
                model_filename = base_model_filename
                model_path = os.path.join('models', model_filename)

                if not os.path.exists(model_path):
                    print(f"Warning: Model file not found at {model_path}. Skipping.")
                    continue
                
                print(f"--- Evaluating model: {model_filename} ---")

                if strategy_name not in test_data:
                    print(f"Warning: No test data found for strategy '{strategy_name}'. Skipping.")
                    continue
                
                test_df = test_data[strategy_name].copy()
                target_col = f'target_{direction}'

                if target_col not in test_df.columns:
                    print(f"Warning: Target column {target_col} not found for {model_filename}. Skipping.")
                    continue
                    
                y_test = test_df.pop(target_col)
                # Drop identifiers and other targets to create the feature set
                X_test = test_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

                for threshold in backtest_thresholds:
                    metrics = evaluate_model(model_path, X_test, y_test, threshold=threshold)
                    
                    results.append({
                        'Model Name': model_filename,
                        'Strategy': strategy_name.capitalize(),
                        'Direction': direction.capitalize(),
                        'Algorithm': model_type.replace('_', ' ').title(),
                        'Threshold': threshold,
                        **metrics
                    })

    # --- Display Consolidated Report ---
    if not results:
        print("No results to display.")
        return
        
    results_df = pd.DataFrame(results)
    
    # --- Save Report to File ---
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    base_report_path = os.path.join(output_dir, 'classification_metrics.csv')
    report_path = base_report_path
    results_df.to_csv(report_path, index=False)
    print(f"\nClassification report saved to {report_path}")

    # Format for better readability in console
    for col in ['Precision', 'Recall', 'F1-Score']:
        results_df[col] = results_df[col].apply(lambda x: f"{x:.2%}")
        
    print("\n\n--- Consolidated Model Performance Report ---")
    # A cleaner print of the dataframe
    print(results_df.to_string())

    print("\n--- Model Evaluation Finished ---")

if __name__ == "__main__":
    main()
