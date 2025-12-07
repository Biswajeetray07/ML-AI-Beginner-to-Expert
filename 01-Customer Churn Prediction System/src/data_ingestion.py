import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml
import logging

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok = True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data Loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""
    try:
        df.drop(columns = ['customer_id'], inplace = True)
        logger.debug('Data Preprocessing Completed')
        return df
    except KeyError as e:
        logger.error('Missing expected column in the data: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during data preprocessing: %s', e)
        raise




def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index = False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'),index = False)
        logger.debug('Train and Test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Failed to save data: %s',e)
        raise

def main():
    try:
        params = load_params(params_path="01-Customer Churn Prediction System\src\params.yaml")
        test_size = params['data_ingestion']['test_size']
        data_path = "https://raw.githubusercontent.com/Biswajeetray07/Datasets/refs/heads/main/Bank%20Customer%20Churn%20Prediction.csv"
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size = test_size, random_state=42)
        save_data(train_data, test_data, data_path = './data')
        logger.info('Data Ingestion Completed Successfully')
    except Exception as e:
        logger.error('Data Ingestion Failed: %s', e)
        print(f'Error: {e}')

if __name__ == "__main__":
    main()