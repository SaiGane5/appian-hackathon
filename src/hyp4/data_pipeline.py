# data_pipeline.py
"""
Load raw CSVs and apply shared FeatureEngineer.
Returns X, y for training and X_test plus IDs for submission.
"""
import pandas as pd
from data_loader import load_data
from feature_engineering import FeatureEngineer

def load_and_engineer(train_path='train.csv', test_path='test.csv'):
    # 1. Load raw
    train_df = load_data(train_path, is_train=True)
    test_df  = load_data(test_path,  is_train=False)

    # 2. Engineer features
    fe = FeatureEngineer(visualize=False)
    train_eng = fe.fit_transform(train_df)
    test_eng  = fe.transform(test_df)

    # 3. Split out IDs and target
    X      = train_eng.drop(['ID','Target'], axis=1)
    y      = train_eng['Target']
    X_test = test_eng.drop('ID', axis=1)
    ids    = test_eng['ID']

    return X, y, X_test, ids
