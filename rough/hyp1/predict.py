import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any

from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                            f1_score, confusion_matrix, classification_report,
                            average_precision_score, roc_curve, auc)
from sklearn.model_selection import train_test_split

# Import local modules
from utils import setup_logger, plot_confusion_matrix, create_output_dir
from data import load_data, check_missing_values, check_data_leakage
from features import preprocess
from model import (train_final_model, evaluate_model, plot_feature_importance, 
                 plot_shap_summary, find_optimal_threshold)

# Configure logger
logger = setup_logger(__name__)

def generate_evaluation_report(model: Any, X: pd.DataFrame, y: pd.Series, 
                              threshold: float, output_dir: str) -> None:
    """
    Generate and save comprehensive evaluation report.
    
    Args:
        model: Trained model
        X: Feature dataframe
        y: Target series
        threshold: Classification threshold
        output_dir: Directory to save outputs
    """
    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = evaluate_model(model, X, y, threshold)
    
    # Save textual report
    with open(os.path.join(output_dir, 'model_report.txt'), 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("======================\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_report(y, y_pred)}\n\n")
        f.write(f"Optimal Threshold: {threshold:.4f}\n")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(metrics['confusion_matrix'], classes=['Not Purchase', 'Purchase'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot and save precision-recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y, y_proba)
    avg_precision = average_precision_score(y, y_proba)
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.axhline(y=sum(y)/len(y), color='red', linestyle='--', 
                label=f'Baseline (class ratio = {sum(y)/len(y):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Save threshold optimization curve
    plt.figure(figsize=(10, 6))
    f1_scores = []
    thresholds = np.linspace(0.1, 0.9, 100)
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y, y_pred_thresh))
    
    plt.plot(thresholds, f1_scores, '-o', markersize=3)
    plt.axvline(x=threshold, color='red', linestyle='--', 
                label=f'Optimal threshold = {threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_optimization.png'))
    plt.close()
    
    logger.info(f"Evaluation report saved to {output_dir}")

def main() -> None:
    """
    Main function to run the prediction pipeline.
    """
    # Create output directory
    output_dir = create_output_dir("outputs")
    
    # Define paths
    data_dir = os.path.join('..', 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    sample_path = os.path.join(data_dir, 'sample_submission.csv')
    
    # Load data
    logger.info("Loading data...")
    train_df, val_df, test_df, sample = load_data(
        train_path, test_path, sample_path, val_size=0.2
    )
    
    # Check data quality
    logger.info("Checking data quality...")
    train_missing = check_missing_values(train_df)
    test_missing = check_missing_values(test_df)
    
    if not train_missing.empty:
        logger.warning(f"Missing values in training data:\n{train_missing}")
    
    if not test_missing.empty:
        logger.warning(f"Missing values in test data:\n{test_missing}")
    
    check_data_leakage(train_df, test_df)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X_train, y_train, train_ids, fit_columns, scalers = preprocess(train_df, is_train=True)
    
    if val_df is not None:
        X_val, y_val, val_ids, _, _ = preprocess(val_df, is_train=True)
    else:
        X_val, y_val = None, None
    
    X_test, test_ids = preprocess(test_df, is_train=False, fit_columns=fit_columns, scalers=scalers)
    
    # Train model
    logger.info("Training final model...")
    model, threshold, selected_features = train_final_model(
        X_train, y_train, 
        n_trials=30,
        use_feature_selection=True,
        n_features=50, 
        calibrate_threshold=True
    )
    
    # Save the model
    model_path = os.path.join(output_dir, 'final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': threshold,
            'selected_features': selected_features,
            'fit_columns': fit_columns
        }, f)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate model on training set
    logger.info("Evaluating model on training set...")
    train_metrics = evaluate_model(model, X_train[selected_features], y_train, threshold)
    logger.info(f"Training ROC AUC: {train_metrics['roc_auc']:.4f}")
    logger.info(f"Training F1 Score: {train_metrics['f1']:.4f}")
    
    # Evaluate model on validation set if available
    if X_val is not None and y_val is not None:
        logger.info("Evaluating model on validation set...")
        val_metrics = evaluate_model(model, X_val[selected_features], y_val, threshold)
        logger.info(f"Validation ROC AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"Validation F1 Score: {val_metrics['f1']:.4f}")
        
        # Generate detailed evaluation report
        generate_evaluation_report(model, X_val[selected_features], y_val, threshold, output_dir)
    
    # Make predictions on test set
    logger.info("Making predictions on test set...")
    test_probs = model.predict_proba(X_test[selected_features])[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'Target': test_preds
    })
    
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
    
    # Save feature importance plots
    try:
        logger.info("Generating feature importance plots...")
        plt.figure(figsize=(12, 10))
        plot_feature_importance(model, X_train[selected_features], top_n=20)
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        plot_shap_summary(model, X_train[selected_features], top_n=20)
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        plt.close()
    except Exception as e:
        logger.warning(f"Error generating feature importance plots: {e}")
    
    logger.info("Processing complete!")

if __name__ == '__main__':
    main()