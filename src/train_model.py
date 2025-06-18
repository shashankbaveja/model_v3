import pandas as pd
import yaml
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from catboost import CatBoostClassifier

# Add the root directory for local library import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import load_config


def load_data(strategy, target_name, config):
    """Loads feature and target data for a given strategy and interval."""
    print(f"Loading data for strategy: {strategy}, target: {target_name}")
    
    processed_dir = 'data/processed'
    
    base_train_filename = f'train_{strategy}_with_patterns_features.parquet'
    train_filename = os.path.join(processed_dir, base_train_filename)
    print(f"Train filename: {train_filename}")

    base_val_filename = f'validation_{strategy}_with_patterns_features.parquet'
    val_filename = os.path.join(processed_dir, base_val_filename)
    
    try:
        train_df = pd.read_parquet(train_filename)

        y_train = train_df.pop(target_name)
        X_train = train_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

        val_df = pd.read_parquet(val_filename)
        y_val = val_df.pop(target_name)
        X_val = val_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Have you run feature generation and merging? Details: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Target column not found. Details: {e}")
        sys.exit(1)

    return X_train, y_train, X_val, y_val

def get_model(model_type, random_state, scale_pos_weight):
    """Returns a model instance based on the type."""
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(random_state=random_state, scale_pos_weight=scale_pos_weight)
    
    elif model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1)
        
    elif model_type == "logistic_regression":
        # Logistic Regression benefits from feature scaling
        return Pipeline([
            ('scaler', StandardScaler()),
            ('logit', LogisticRegression(random_state=random_state, class_weight='balanced', solver='liblinear'))
        ])
    elif model_type == "catboost":
        return CatBoostClassifier(random_state=random_state, scale_pos_weight=scale_pos_weight, verbose=0)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(strategy, target_name, model_type, config):
    """Trains a single model for a given strategy, target, and model type."""
    model_config = config['modeling']
    random_state = model_config['random_state']
    
    X_train, y_train, X_val, y_val = load_data(strategy, target_name, config)
    
    print(f"Training a {model_type} model...")
    print(f"  Training data loaded. Number of features: {len(X_train.columns)}")

    # Calculate scale_pos_weight for LightGBM
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    model = get_model(model_type, random_state, scale_pos_weight)
    
    # Fit the model
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Save the model
    direction = 'up' if 'up' in target_name else 'down'
    base_model_filename = f'{strategy}_{direction}_{model_type}_model.joblib'
    model_filename = base_model_filename
    
    model_path = os.path.join('models', model_filename)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

def main():
    """Main function to run the model training pipeline."""
    print("--- Starting Model Training Pipeline ---")
    config = load_config('config/parameters.yml')
    
    # Get parameters from the unified config
    model_config = config.get('modeling', {})
    model_types = model_config.get('model_types', ['lightgbm'])
    strategies = model_config.get('strategies_to_train', ['momentum', 'reversion','combined'])
    targets = model_config.get('targets', ['target_up', 'target_down'])
    
    for strategy in strategies:
        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} Models for {strategy.upper()} Strategy ---")
            for target_name in targets:
                train_model(strategy, target_name, model_type, config)

    print("\n--- Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()
