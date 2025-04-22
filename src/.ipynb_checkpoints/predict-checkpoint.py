import os
from utils import setup_logger
from data import load_data
from features import preprocess
from model import tune_hyperparameters, train_final_model

logger = setup_logger(__name__)

def main():
    # Paths
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    sample_path = os.path.join('data', 'sample_submission.csv')

    # Load
    train_df, test_df, sample = load_data(train_path, test_path, sample_path)
    logger.info('Data loaded.')

    # Preprocess
    X_train, y_train, _ = preprocess(train_df, is_train=True)
    X_test, test_ids = preprocess(test_df, is_train=False)
    logger.info('Preprocessing complete.')

    # Tune
    best_params = tune_hyperparameters(X_train, y_train)
    logger.info(f'Best params: {best_params}')

    # Train final
    model = train_final_model(X_train, y_train, best_params)

    # Predict
    preds = model.predict_proba(X_test)[:, 1]
    sample['Target'] = preds
    sample.to_csv('submission.csv', index=False)
    logger.info('Submission saved to submission.csv')

if __name__ == '__main__':
    main()