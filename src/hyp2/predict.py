# predict.py
from data_loader import load_data
from feature_engineering import FeatureEngineerer
import pandas as pd
import joblib
import json

def main():
    # Load test data
    test_df = load_data('test.csv', is_train=False)
    test_ids = test_df['ID']
    
    # Load engineered features schema
    with open('feature_columns.json') as f:
        feature_columns = json.load(f)
    
    # Feature engineering
    engineer = FeatureEngineerer()
    test_engineered = engineer.transform(test_df)
    
    # Column alignment
    missing_cols = set(feature_columns) - set(test_engineered.columns)
    for col in missing_cols:
        test_engineered[col] = 0
    test_engineered = test_engineered[feature_columns]
    
    # Load model and predict
    model = joblib.load('rf_model.pkl')
    predictions = model.predict(test_engineered)
    
    # Save submission
    pd.DataFrame({'ID': test_ids, 'Target': predictions})\
      .to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
