import os
from utils import setup_logger
from data import load_data
from features import preprocess
from model import train_final_model
from sklearn.metrics import precision_recall_curve
import numpy as np
logger = setup_logger(__name__)

def main():
    # Paths
    train_path = os.path.join('../data', 'train.csv')
    test_path = os.path.join('../data', 'test.csv')
    sample_path = os.path.join('../data', 'sample_submission.csv')

    # Load
    train_df, test_df, sample = load_data(train_path, test_path, sample_path)
    logger.info('Data loaded.')

    # Preprocess
    X_train, y_train, _, fit_columns = preprocess(train_df, is_train=True)
    X_test, test_ids = preprocess(test_df, is_train=False, fit_columns=fit_columns)
    logger.info('Preprocessing complete.')

    # Train final
    model = train_final_model(X_train, y_train)
    logger.info('Model trained.')
    
    y_proba = model.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Use optimal threshold for test predictions
    test_probs = model.predict_proba(X_test)[:, 1]
    preds = (test_probs >= optimal_threshold).astype(int)
    # Predict
    preds = model.predict(X_test)  # Changed from predict_proba
    sample['Target'] = preds.astype(int)  # Ensure integer output
    sample.to_csv('submission.csv', index=False)
    logger.info('Submission saved to submission.csv')

if __name__ == '__main__':
    main()