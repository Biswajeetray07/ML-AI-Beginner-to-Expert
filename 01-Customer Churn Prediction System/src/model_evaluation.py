import os
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import yaml
import json
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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

def load_model(file_path: str):
    """Load the trained model from pickle file"""

    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f'Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('Model file not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return evaluation metrics"""
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc':auc
        }
        logger.debug('Model evaluation completed with metrics: %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error('Unexpected error occurred during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save evaluation ,metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok = True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent = 4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Failed to save metrics: %s', e)
        raise

def main():
    """Main function to execute model evealuation."""
    try:
        params = load_params(params_path='01-Customer Churn Prediction System\src\params.yaml')
        test_data = load_data('./data/processed/test_fe.csv')

        logger.debug('Test Data Loaded for Model Evaluation')
        x_test = test_data.drop(columns = ['churn']).values
        y_test = test_data['churn'].values
    

        model_files = {
            "RandomForest": './models/random_forest_model.pkl',
            "LogisticRegression": './models/logistic_regression_model.pkl',
            "XGBoost": './models/xgboost_model.pkl'
        }
        metrics = {}
        for model_files,file_path in model_files.items():
            model = load_model(file_path)
            model_metrics = evaluate_model(model, x_test, y_test)
            metrics[model_files] = model_metrics

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        save_metrics(metrics, './reports/model_evaluation_metrics.json')
    except Exception as e:
        logger.error('Failed to complete model evaluation process: %s', e)
        print(f'Error:{e}')

if __name__ == '__main__':
    main()