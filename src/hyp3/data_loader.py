import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_loader')

def create_debug_dir():
    """Create debug directory if it doesn't exist"""
    debug_dir = 'debug'
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    return debug_dir

def load_data(file_path, is_train=True):
    """
    Load and preprocess data with proper type specification and debugging
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Define dtype mapping for memory efficiency and type safety
        dtype_mapping = {
            'ID': 'int32',
            'Year_Birth': 'int32',
            'Education': 'category',
            'Marital_Status': 'category',
            'Income': 'float32',
            'Kidhome': 'int8',
            'Teenhome': 'int8',
            'Dt_Customer': 'object',  # Will parse to datetime after loading
            'Recency': 'int16',
            'Complain': 'int8',
            'MntWines': 'float32',
            'MntFruits': 'float32',
            'MntMeatProducts': 'float32',
            'MntFishProducts': 'float32',
            'MntSweetProducts': 'float32',
            'MntGoldProds': 'float32',
            'NumDealsPurchases': 'int8',
            'AcceptedCmp1': 'int8',
            'AcceptedCmp2': 'int8',
            'AcceptedCmp3': 'int8',
            'AcceptedCmp4': 'int8',
            'AcceptedCmp5': 'int8',
            'NumWebPurchases': 'int8',
            'NumCatalogPurchases': 'int8',
            'NumStorePurchases': 'int8',
            'NumWebVisitsMonth': 'int8'
        }
        
        # Add Target column dtype if this is a training file
        if is_train:
            dtype_mapping['Target'] = 'int8'
        
        # Read CSV with proper error handling
        df = pd.read_csv(
            file_path,
            dtype=dtype_mapping,
            na_values=['', 'unknown', 'Unknown', 'NA', 'na', 'N/A', 'n/a']
        )
        
        # Log initial data stats
        logger.info(f"Initial data shape: {df.shape}")
        logger.info(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Debug output - save a copy of raw data
        debug_dir = create_debug_dir()
        fname = os.path.basename(file_path).split('.')[0]
        df.head(10).to_csv(f"{debug_dir}/{fname}_raw_sample.csv")
        
        # Check and log missing values
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if len(missing_values) > 0:
            logger.warning(f"Missing values detected:\n{missing_values}")
        
        # Convert date column safely
        logger.info("Converting Dt_Customer to datetime")
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
        
        # Handle missing dates - use median date if any missing
        if df['Dt_Customer'].isnull().sum() > 0:
            median_date = df['Dt_Customer'].dropna().median()
            logger.warning(f"Found {df['Dt_Customer'].isnull().sum()} missing dates. Filling with median: {median_date}")
            df['Dt_Customer'] = df['Dt_Customer'].fillna(median_date)
        
        # Create numeric feature for customer registration duration
        logger.info("Creating Days_Since_Registration feature")
        reference_date = datetime(2025, 4, 23)  # Using current date from prompt
        df['Days_Since_Registration'] = (reference_date - df['Dt_Customer']).dt.days
        df = df.drop('Dt_Customer', axis=1)
        
        # Handle missing values in Income
        if df['Income'].isnull().sum() > 0:
            logger.info(f"Filling {df['Income'].isnull().sum()} missing Income values with median")
            df['Income'] = df['Income'].fillna(df['Income'].median())
        
        # Handle categorical features
        for col in ['Education', 'Marital_Status']:
            logger.info(f"Processing categorical column: {col}")
            
            # Add 'Unknown' category if not present and fill missing values
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('Unknown')
            
            if df[col].isnull().sum() > 0:
                logger.info(f"Filling {df[col].isnull().sum()} missing {col} values with 'Unknown'")
                df[col] = df[col].fillna('Unknown')
        
        # Clean numeric columns - replace negatives with 0 for counts
        count_columns = ['Kidhome', 'Teenhome', 'Recency', 'NumDealsPurchases', 
                         'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                         'NumWebVisitsMonth']
        
        for col in count_columns:
            if (df[col] < 0).sum() > 0:
                logger.warning(f"Found {(df[col] < 0).sum()} negative values in {col}. Setting to 0.")
                df[col] = df[col].clip(lower=0)
        
        # Clean monetary columns - replace negatives with 0 for amounts
        monetary_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        
        for col in monetary_columns:
            if (df[col] < 0).sum() > 0:
                logger.warning(f"Found {(df[col] < 0).sum()} negative values in {col}. Setting to 0.")
                df[col] = df[col].clip(lower=0)
        
        # Handle outliers in Income - cap at 99.5th percentile
        if 'Income' in df.columns:
            cap_value = df['Income'].quantile(0.995)
            outliers = (df['Income'] > cap_value).sum()
            if outliers > 0:
                logger.warning(f"Capping {outliers} outliers in Income at {cap_value}")
                df['Income'] = df['Income'].clip(upper=cap_value)
        
        # Handle extreme ages - cap at reasonable values
        if 'Year_Birth' in df.columns:
            current_year = datetime.now().year
            min_age, max_age = 18, 100
            min_year, max_year = current_year - max_age, current_year - min_age
            
            # Check for unreasonable years
            unreasonable_years = ((df['Year_Birth'] < min_year) | (df['Year_Birth'] > max_year)).sum()
            if unreasonable_years > 0:
                logger.warning(f"Found {unreasonable_years} unreasonable birth years. Capping between {min_year} and {max_year}")
                df['Year_Birth'] = df['Year_Birth'].clip(lower=min_year, upper=max_year)
        
        # Ensure binary columns are truly binary
        binary_columns = ['Complain', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                         'AcceptedCmp4', 'AcceptedCmp5']
        
        if is_train and 'Target' in df.columns:
            binary_columns.append('Target')
        
        for col in binary_columns:
            if col in df.columns:
                unique_vals = df[col].unique()
                if not all(val in [0, 1, np.nan] for val in unique_vals):
                    logger.warning(f"Column {col} contains non-binary values: {unique_vals}")
                    df[col] = df[col].map(lambda x: 1 if x > 0 else 0)
        
        # Check for duplicated IDs
        if df['ID'].duplicated().sum() > 0:
            logger.warning(f"Found {df['ID'].duplicated().sum()} duplicate IDs")
            # Keep first occurrence of each ID
            df = df.drop_duplicates(subset=['ID'], keep='first')
        
        # Final verification - check for any remaining nulls
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.warning(f"There are still {remaining_nulls} missing values in the dataset")
            # Fill any remaining nulls with appropriate values
            df = df.fillna(df.median(numeric_only=True))
        
        # Log final data stats
        logger.info(f"Final data shape: {df.shape}")
        logger.info(f"Missing values after preprocessing: {df.isnull().sum().sum()}")
        
        # Debug output - save preprocessed data sample
        df.head(10).to_csv(f"{debug_dir}/{fname}_preprocessed_sample.csv")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise
