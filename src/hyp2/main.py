from data_loader import load_data
from feature_engineering import FeatureEngineerer
from eda import run_eda

def main():
    # Load and preprocess data
    train_df = load_data('train.csv')
    
    # Perform initial EDA on raw data
    # run_eda(train_df)
    
    # Feature engineering
    engineer = FeatureEngineerer()
    train_engineered = engineer.fit_transform(train_df)
    
    # EDA on engineered features
    run_eda(train_engineered)
    
    # Save processed data
    train_engineered.to_parquet('processed_train.parquet', index=False)

if __name__ == '__main__':
    main()
