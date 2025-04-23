from data_loader import load_data
from feature_engineering import FeatureEngineerer
from eda import run_eda
from train_model import train_and_save_model

def main():
    # Load and preprocess training data
    train_df = load_data('train.csv')
    run_eda(train_df)
    engineer = FeatureEngineerer()
    train_engineered = engineer.fit_transform(train_df)
    train_engineered.to_csv('processed_train.csv', index=False)
    run_eda(train_engineered)
    # Train and save model
    model, feature_columns = train_and_save_model(train_engineered, model_path='rf_model.pkl')
    # Save feature columns for prediction step
    import json
    with open('feature_columns.json', 'w') as f:
        json.dump(list(feature_columns), f)

if __name__ == '__main__':
    main()
