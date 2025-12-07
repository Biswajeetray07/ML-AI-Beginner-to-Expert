import os
import numpy as np
import pandas as pd
import logging
import pickle
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data Loaded for Model Building from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> Tuple[RandomForestClassifier, LogisticRegression, XGBClassifier]:
    """Train multiple models and return them"""
    try:
        sm = SMOTE(random_state=42)
        x_res, y_res = sm.fit_resample(x_train, y_train)
        logger.debug('Applied SMOTE to balance the classes in training data')

        rf_model = RandomForestClassifier(**params.get('random_forest', {}))
        lr_model = LogisticRegression(**params.get('logistic_regression', {}))
        xgb_model = XGBClassifier(**params.get('xgboost', {}))

        logger.debug('Model training started with %d training samples after resampling', x_res.shape[0])

        rf_model.fit(x_res, y_res)
        lr_model.fit(x_res, y_res)
        xgb_model.fit(x_res, y_res)

        logger.debug('Model training completed')
        return rf_model, lr_model, xgb_model
    except KeyError as e:
        logger.error('Missing expected parameter for model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred during model training: %s', e)
        raise

def save_model(model, file_path:str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File not found for saving the model: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to save model: %s', e)
        raise

def main():
    """Main function to execute model building."""
    try:
        params = load_params(params_path='01-Customer Churn Prediction System\src\params.yaml')
        train_data = load_data('./data/processed/train_fe.csv')
        logger.debug('Train and Test Data Loaded for Model Building')
        x_train = train_data.drop(columns=['churn']).values
        y_train = train_data['churn'].values
        rf_model, lr_model, xgb_model = train_model(x_train, y_train, params)

        logger.debug('Saving trained models')
        save_model(rf_model, './models/random_forest_model.pkl')
        save_model(lr_model, './models/logistic_regression_model.pkl')
        save_model(xgb_model, './models/xgboost_model.pkl')
        logger.debug('All models saved successfully')
    except Exception as e:
        logger.error('Failed to complete the model building pipeline: %s', e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()

