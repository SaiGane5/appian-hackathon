import pandas as pd
import os
from typing import Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(train_path: str, test_path: str, sample_path: str, 
              val_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Load train, test, and sample submission with validation split capability.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        sample_path: Path to sample submission
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing train, validation, test, and sample submission dataframes
    """
    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    
    # Create validation set
    if val_size > 0:
        train, val = train_test_split(train, test_size=val_size, 
                                     random_state=random_state, 
                                     stratify=train['Target'])
        return train, val, test, sample
    
    return train, None, test, sample

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with missing value statistics
    """
    missing = pd.DataFrame({
        'count': df.isnull().sum(),
        'percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    return missing[missing['count'] > 0].sort_values('percentage', ascending=False)

def check_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Check for potential data leakage between train and test sets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
    """
    # Check for ID overlap
    train_ids = set(train_df['ID'])
    test_ids = set(test_df['ID'])
    overlap = train_ids.intersection(test_ids)
    
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping IDs between train and test")
    else:
        print("No ID overlap between train and test sets")