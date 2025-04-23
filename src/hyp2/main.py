import os
import logging
import argparse
import pandas as pd
import joblib
import json
from datetime import datetime

# Import all components
from data_loader import load_data
from feature_engineering import FeatureEngineer
from eda import run_eda
from train_model import ModelTrainer
from predict import ModelPredictor
from model_evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

def setup_directories():
    """Create necessary directories for the pipeline"""
    dirs = ['models', 'results', 'debug', 'submissions']
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def run_full_pipeline(train_path='train.csv', test_path='test.csv', 
                    submission_path='submissions/submission.csv',
                    perform_eda=True, save_all_models=False):
    """
    Run the complete machine learning pipeline
    
    Parameters:
    -----------
    train_path : str
        Path to training data
    test_path : str
        Path to test data
    submission_path : str
        Path to save final submission
    perform_eda : bool
        Whether to perform exploratory data analysis
    save_all_models : bool
        Whether to save all trained models or just the best one
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting Appian product purchase prediction pipeline at {timestamp}")
    
    # Setup directories
    setup_directories()
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    train_df = load_data(train_path, is_train=True)
    test_df = load_data(test_path, is_train=False)
    
    # Save IDs for later submission
    test_ids = test_df['ID'].copy()
    
    # Step 2: Exploratory Data Analysis (if requested)
    if perform_eda:
        logger.info("Step 2: Performing Exploratory Data Analysis")
        run_eda(train_df)
    
    # Step 3: Feature Engineering
    logger.info("Step 3: Engineering features")
    engineer = FeatureEngineer()
    train_engineered = engineer.fit_transform(train_df)
    test_engineered = engineer.transform(test_df)
    
    # Save processed data for reference
    train_engineered.to_csv('debug/processed_train_sample.csv', index=False)
    test_engineered.head().to_csv('debug/processed_test_sample.csv', index=False)
    
    # Step 4: Train models
    logger.info("Step 4: Training models")
    X = train_engineered.drop(['ID', 'Target'], axis=1, errors='ignore')
    y = train_engineered['Target']
    
    trainer = ModelTrainer(model_dir='models', results_dir='results')
    results = trainer.train(X, y, test_size=0.2, cv=5, 
                          scoring='roc_auc', find_best=True, 
                          save_all=save_all_models)
    
    best_model = trainer.best_model
    best_model_name = trainer.best_model_name
    feature_columns = trainer.feature_columns
    
    logger.info(f"Best model: {best_model_name}")
    
    # Step 5: Evaluate best model
    logger.info("Step 5: Evaluating best model")
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)
    
    evaluator = ModelEvaluator(results_dir='results')
    evaluation_results = evaluator.evaluate_and_save(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        model_name=best_model_name
    )
    
    # Step 6: Make predictions on test data
    logger.info("Step 6: Making predictions on test data")
    X_test_final = test_engineered.drop(['ID'], axis=1, errors='ignore')
    
    # Align columns with training data
    missing_cols = set(feature_columns) - set(X_test_final.columns)
    for col in missing_cols:
        X_test_final[col] = 0
    
    X_test_final = X_test_final[feature_columns]
    
    # Make predictions
    y_pred = best_model.predict(X_test_final)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'Target': y_pred.astype(int)
    })
    
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    
    # Save submission
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
    
    # Step 7: Create experiment tracking record
    experiment_info = {
        'timestamp': timestamp,
        'best_model': best_model_name,
        'feature_count': len(feature_columns),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'positive_predictions': int(sum(y_pred)),
        'positive_rate': float(sum(y_pred) / len(y_pred)),
        'best_metrics': {k: float(v) for k, v in evaluation_results['test_results']['metrics'].items() 
                        if not isinstance(v, (list, dict, pd.DataFrame, pd.Series))}
    }
    
    # Save experiment info
    with open(f'results/experiment_{timestamp}.json', 'w') as f:
        json.dump(experiment_info, f, indent=4)
    
    logger.info("Pipeline completed successfully")
    return submission, best_model, experiment_info

def run_prediction_only(test_path='test.csv', model_path='models/best_model.pkl',
                      submission_path='submissions/submission.csv'):
    """
    Run only the prediction part of the pipeline
    
    Parameters:
    -----------
    test_path : str
        Path to test data
    model_path : str
        Path to trained model
    submission_path : str
        Path to save final submission
    """
    logger.info(f"Starting prediction-only pipeline")
    
    # Setup directories
    setup_directories()
    
    # Load test data
    test_df = load_data(test_path, is_train=False)
    
    # Make predictions
    predictor = ModelPredictor()
    submission = predictor.predict(
        test_df=test_df,
        model_path=model_path,
        submission_filename=submission_path
    )
    
    logger.info(f"Prediction completed. Submission saved to {submission_path}")
    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Appian Product Purchase Prediction Pipeline')
    
    parser.add_argument('--mode', type=str, default='full',
                      choices=['full', 'predict'],
                      help='Pipeline mode: full or predict-only')
    
    parser.add_argument('--train', type=str, default='train.csv',
                      help='Path to training data')
    
    parser.add_argument('--test', type=str, default='test.csv',
                      help='Path to test data')
    
    parser.add_argument('--model', type=str, default='models/best_model.pkl',
                      help='Path to trained model (for predict mode)')
    
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                      help='Path to save submission file')
    
    parser.add_argument('--skip-eda', action='store_true',
                      help='Skip exploratory data analysis')
    
    parser.add_argument('--save-all-models', action='store_true',
                      help='Save all trained models, not just the best one')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline(
            train_path=args.train,
            test_path=args.test,
            submission_path=args.output,
            perform_eda=not args.skip_eda,
            save_all_models=args.save_all_models
        )
    elif args.mode == 'predict':
        run_prediction_only(
            test_path=args.test,
            model_path=args.model,
            submission_path=args.output
        )
