import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
        logger.debug(f'Data Loaded for Preprocessing from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""
    try:
        # Scaling Credit_Score column
        df['credit_score'] = StandardScaler().fit_transform(df[['credit_score']])

        # One-Hot Encoding categorical columns
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(df[['country']])
        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['country']))
        # Join back to main df
        df = pd.concat([df.drop(columns=['country']), encoded_df], axis=1)

        # Mapping Gender column
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # Scaling Products_Number column
        df[['products_number','estimated_salary']] = StandardScaler().fit_transform(df[['products_number', 'estimated_salary']])

        logger.debug('Data Preprocessing Completed')
        return df
    except KeyError as e:
        logger.error('Missing expected column in the data: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during data preprocessing: %s', e)
        raise

def main():
    """Main function to execute data preprocessing."""
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Train and Test Data Loaded for Preprocessing')

        # Preprocess the data
        train_processed_data = preprocess_df(train_data)
        test_processed_data = preprocess_df(test_data)

        # Save the processed data
        data_path = os.path.join('./data', "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug('Processed Train and Test data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data preprocessing pipeline: %s', e)
        print(f'Error: {e}')

if __name__ == "__main__":
    main()