import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data Loaded for feature engineering from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def feature_engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering the data"""
    try:
        # Scaling Age column
        df['age_original'] = df['age']
        df['age'] = StandardScaler().fit_transform(df[['age']])
        df['age_bucket'] = df['age_original'].apply(lambda x: 'young' if x < 30 else 'adult' if x <= 50 else 'senior')
        df = pd.get_dummies(df, columns=['age_bucket'], prefix='age')
        df.drop(columns=['age_original'], inplace=True)

        # Scaling tenure column
        df['tenure_original'] = df['tenure']
        df['tenure'] = StandardScaler().fit_transform(df[['tenure']])
        df['tenure_bucket'] = df['tenure_original'].apply(lambda x: 'short_term' if x <= 3 else 'medium_term' if x <= 7  else 'long_term' )
        df = pd.get_dummies(df, columns=['tenure_bucket'], prefix = 'tenure')
        df.drop(columns = ['tenure_original'], inplace = True)

        # Scaling Balance column
        df['balance_original'] = df['balance']
        df['balance'] = StandardScaler().fit_transform(df[['balance']])
        df['balance_to_salary_ratio'] = df['balance_original']/df['estimated_salary']
        df['balance_to_salary_ratio'] = StandardScaler().fit_transform(df[['balance_to_salary_ratio']])
        df.drop(columns = ['balance_original'], inplace = True)

        logger.debug('Data Preprocessing Completed')
        return df
    except KeyError as e:
        logger.error('Missing expected column in the data: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during data preprocessing: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        df.to_csv(file_path, index = False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Failed to save data: %s',e)
        raise

def main():
    try:
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Feature engineer the data
        train_fe_data, test_fe_data = feature_engineer_data(train_data), feature_engineer_data(test_data)

        save_data(train_fe_data, os.path.join("./data","processed","train_fe.csv"))
        save_data(test_fe_data, os.path.join("./data","processed","test_fe.csv"))
        logger.debug('Feature engineered Train and Test data saved successfully')
    except Exception as e:
        logger.error('Error in feature engineering process: %s', e)
        print(f'Error:{e}')

if __name__ == '__main__':
    main()        