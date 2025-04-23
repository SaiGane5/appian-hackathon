import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_evaluation')

class ModelEvaluator:
    """
    Comprehensive model evaluation for Appian purchase prediction.
    Provides detailed metrics, visualizations, and cross-validation analysis.
    """
    
    def __init__(self, results_dir='results'):
        """
        Initialize the model evaluator
        
        Parameters:
        -----------
        results_dir : str
            Directory to save evaluation results
        """
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'evaluation_plots')
        
        # Create directories if they don't exist
        for directory in [self.results_dir, self.plots_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """
        Evaluate model on test data with various metrics
        
        Parameters:
        -----------
        model : estimator
            Trained model
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_name} on test data")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            has_proba = hasattr(model, 'predict_proba')
            if has_proba:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # Calculate standard classification metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred)
            }
            
            # Add probability-based metrics if available
            if has_proba:
                metrics.update({
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'average_precision': average_precision_score(y_test, y_pred_proba)
                })
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Get classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log basic metrics
            logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"Test precision: {metrics['precision']:.4f}")
            logger.info(f"Test recall: {metrics['recall']:.4f}")
            logger.info(f"Test F1 score: {metrics['f1']:.4f}")
            
            if has_proba:
                logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
                logger.info(f"Test average precision: {metrics['average_precision']:.4f}")
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['No Purchase', 'Purchase'],
                      yticklabels=['No Purchase', 'Purchase'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_confusion_matrix.png'))
            plt.close()
            
            # Plot ROC curve if probabilities are available
            if has_proba:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.plots_dir, f'{model_name}_roc_curve.png'))
                plt.close()
                
                # Plot precision-recall curve
                plt.figure(figsize=(8, 6))
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                plt.plot(recall, precision, label=f'PR Curve (AP = {metrics["average_precision"]:.4f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {model_name}')
                plt.legend(loc='lower left')
                plt.savefig(os.path.join(self.plots_dir, f'{model_name}_pr_curve.png'))
                plt.close()
                
                # Plot calibration curve
                plt.figure(figsize=(8, 6))
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba, n_bins=10)
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
                plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
                plt.xlabel('Mean predicted probability')
                plt.ylabel('Fraction of positives')
                plt.title(f'Calibration Curve - {model_name}')
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.plots_dir, f'{model_name}_calibration_curve.png'))
                plt.close()
                
                # Plot histogram of predicted probabilities
                plt.figure(figsize=(10, 6))
                
                # Separate predictions by actual class
                pos_probs = y_pred_proba[y_test == 1]
                neg_probs = y_pred_proba[y_test == 0]
                
                # Plot histograms
                plt.hist(neg_probs, bins=20, alpha=0.5, color='blue', label='Actual: No Purchase')
                plt.hist(pos_probs, bins=20, alpha=0.5, color='red', label='Actual: Purchase')
                plt.xlabel('Predicted Probability')
                plt.ylabel('Count')
                plt.title(f'Distribution of Predicted Probabilities - {model_name}')
                plt.legend()
                plt.savefig(os.path.join(self.plots_dir, f'{model_name}_probability_distribution.png'))
                plt.close()
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df['model'] = model_name
            metrics_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            metrics_file = os.path.join(self.results_dir, 'model_metrics.csv')
            
            # Append to existing file if it exists
            if os.path.exists(metrics_file):
                existing_metrics = pd.read_csv(metrics_file)
                updated_metrics = pd.concat([existing_metrics, metrics_df], ignore_index=True)
                updated_metrics.to_csv(metrics_file, index=False)
            else:
                metrics_df.to_csv(metrics_file, index=False)
            
            # Return the metrics
            result = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': report,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
            raise
    
    def cross_validate(self, model, X, y, cv=5, model_name='model'):
        """
        Perform cross-validation evaluation
        
        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : pandas.DataFrame
            Feature data
        y : pandas.Series
            Target data
        cv : int
            Number of cross-validation folds
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary of cross-validation metrics
        """
        try:
            logger.info(f"Performing {cv}-fold cross-validation for {model_name}")
            
            # Setup cross-validation
            cv_stratified = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Check if model supports predict_proba
            has_proba = hasattr(model, 'predict_proba')
            
            # Get cross-validation predictions
            y_pred = cross_val_predict(model, X, y, cv=cv_stratified)
            
            if has_proba:
                y_pred_proba = cross_val_predict(model, X, y, cv=cv_stratified, method='predict_proba')[:, 1]
            else:
                y_pred_proba = None
            
            # Calculate metrics
            cv_metrics = {
                'cv_accuracy': accuracy_score(y, y_pred),
                'cv_balanced_accuracy': balanced_accuracy_score(y, y_pred),
                'cv_precision': precision_score(y, y_pred),
                'cv_recall': recall_score(y, y_pred),
                'cv_f1': f1_score(y, y_pred),
                'cv_cohen_kappa': cohen_kappa_score(y, y_pred)
            }
            
            # Add probability-based metrics if available
            if has_proba:
                cv_metrics.update({
                    'cv_roc_auc': roc_auc_score(y, y_pred_proba),
                    'cv_average_precision': average_precision_score(y, y_pred_proba)
                })
            
            # Calculate confusion matrix
            cv_cm = confusion_matrix(y, y_pred)
            
            # Get classification report
            cv_report = classification_report(y, y_pred, output_dict=True)
            
            # Log results
            logger.info(f"CV accuracy: {cv_metrics['cv_accuracy']:.4f}")
            logger.info(f"CV balanced accuracy: {cv_metrics['cv_balanced_accuracy']:.4f}")
            logger.info(f"CV precision: {cv_metrics['cv_precision']:.4f}")
            logger.info(f"CV recall: {cv_metrics['cv_recall']:.4f}")
            logger.info(f"CV F1 score: {cv_metrics['cv_f1']:.4f}")
            
            if has_proba:
                logger.info(f"CV ROC AUC: {cv_metrics['cv_roc_auc']:.4f}")
                logger.info(f"CV average precision: {cv_metrics['cv_average_precision']:.4f}")
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cv_cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['No Purchase', 'Purchase'],
                      yticklabels=['No Purchase', 'Purchase'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Cross-Validation Confusion Matrix - {model_name}')
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_cv_confusion_matrix.png'))
            plt.close()
            
            # Plot ROC curve if probabilities are available
            if has_proba:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {cv_metrics["cv_roc_auc"]:.4f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Cross-Validation ROC Curve - {model_name}')
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.plots_dir, f'{model_name}_cv_roc_curve.png'))
                plt.close()
            
            # Save metrics to CSV
            cv_metrics_df = pd.DataFrame([cv_metrics])
            cv_metrics_df['model'] = model_name
            cv_metrics_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv_metrics_df['cv_folds'] = cv
            
            cv_metrics_file = os.path.join(self.results_dir, 'cross_validation_metrics.csv')
            
            # Append to existing file if it exists
            if os.path.exists(cv_metrics_file):
                existing_metrics = pd.read_csv(cv_metrics_file)
                updated_metrics = pd.concat([existing_metrics, cv_metrics_df], ignore_index=True)
                updated_metrics.to_csv(cv_metrics_file, index=False)
            else:
                cv_metrics_df.to_csv(cv_metrics_file, index=False)
            
            # Return the results
            result = {
                'cv_metrics': cv_metrics,
                'cv_confusion_matrix': cv_cm,
                'cv_classification_report': cv_report,
                'cv_predictions': y_pred,
                'cv_probabilities': y_pred_proba
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}", exc_info=True)
            raise
    
    def analyze_threshold(self, y_true, y_pred_proba, model_name='model'):
        """
        Analyze the effect of different classification thresholds
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of the model
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with metrics at different thresholds
        """
        try:
            logger.info(f"Analyzing classification thresholds for {model_name}")
            
            # List of thresholds to test
            thresholds = np.arange(0.1, 1.0, 0.05)
            
            # DataFrame to store results
            results = []
            
            # Calculate metrics for each threshold
            for threshold in thresholds:
                # Convert probabilities to predictions using threshold
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # Calculate metrics
                acc = accuracy_score(y_true, y_pred)
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                # Store results
                results.append({
                    'threshold': threshold,
                    'accuracy': acc,
                    'balanced_accuracy': balanced_acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1
                })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results to CSV
            threshold_file = os.path.join(self.results_dir, f'{model_name}_threshold_analysis.csv')
            results_df.to_csv(threshold_file, index=False)
            
            # Plot metrics vs threshold
            plt.figure(figsize=(12, 8))
            plt.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy')
            plt.plot(results_df['threshold'], results_df['balanced_accuracy'], 'o-', label='Balanced Accuracy')
            plt.plot(results_df['threshold'], results_df['precision'], 'o-', label='Precision')
            plt.plot(results_df['threshold'], results_df['recall'], 'o-', label='Recall')
            plt.plot(results_df['threshold'], results_df['f1'], 'o-', label='F1')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title(f'Classification Metrics vs. Threshold - {model_name}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_threshold_analysis.png'))
            plt.close()
            
            # Find optimal thresholds for different metrics
            optimal_thresholds = {
                'accuracy': results_df.loc[results_df['accuracy'].idxmax(), 'threshold'],
                'balanced_accuracy': results_df.loc[results_df['balanced_accuracy'].idxmax(), 'threshold'],
                'f1': results_df.loc[results_df['f1'].idxmax(), 'threshold']
            }
            
            logger.info(f"Optimal thresholds for {model_name}:")
            logger.info(f"  Accuracy: {optimal_thresholds['accuracy']:.2f}")
            logger.info(f"  Balanced Accuracy: {optimal_thresholds['balanced_accuracy']:.2f}")
            logger.info(f"  F1: {optimal_thresholds['f1']:.2f}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error analyzing thresholds: {e}", exc_info=True)
            raise
    
    def evaluate_and_save(self, model, X_test, y_test, X_train=None, y_train=None, 
                       model_name='model', perform_cv=True, analyze_thresholds=True):
        """
        Comprehensive model evaluation and result saving
        
        Parameters:
        -----------
        model : estimator
            Trained model
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test targets
        X_train : pandas.DataFrame, optional
            Training features (for cross-validation)
        y_train : pandas.Series, optional
            Training targets (for cross-validation)
        model_name : str
            Name of the model
        perform_cv : bool
            Whether to perform cross-validation
        analyze_thresholds : bool
            Whether to analyze classification thresholds
            
        Returns:
        --------
        dict
            Dictionary of all evaluation results
        """
        try:
            logger.info(f"Starting comprehensive evaluation for {model_name}")
            
            # Evaluate on test data
            test_results = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Cross-validation if requested
            cv_results = None
            if perform_cv and X_train is not None and y_train is not None:
                cv_results = self.cross_validate(model, X_train, y_train, cv=5, model_name=model_name)
            
            # Threshold analysis if requested
            threshold_results = None
            if analyze_thresholds and test_results['y_pred_proba'] is not None:
                threshold_results = self.analyze_threshold(
                    y_test, test_results['y_pred_proba'], model_name=model_name)
            
            # Compile all results
            all_results = {
                'test_results': test_results,
                'cv_results': cv_results,
                'threshold_results': threshold_results
            }
            
            # Generate and save overall summary report
            summary = pd.DataFrame({
                'model': [model_name],
                'accuracy': [test_results['metrics']['accuracy']],
                'balanced_accuracy': [test_results['metrics']['balanced_accuracy']],
                'precision': [test_results['metrics']['precision']],
                'recall': [test_results['metrics']['recall']],
                'f1': [test_results['metrics']['f1']]
            })
            
            # Add probability-based metrics if available
            if 'roc_auc' in test_results['metrics']:
                summary['roc_auc'] = test_results['metrics']['roc_auc']
                summary['average_precision'] = test_results['metrics']['average_precision']
            
            # Add CV metrics if available
            if cv_results is not None:
                summary['cv_accuracy'] = cv_results['cv_metrics']['cv_accuracy']
                summary['cv_f1'] = cv_results['cv_metrics']['cv_f1']
                
                if 'cv_roc_auc' in cv_results['cv_metrics']:
                    summary['cv_roc_auc'] = cv_results['cv_metrics']['cv_roc_auc']
            
            # Add optimal thresholds if available
            if threshold_results is not None:
                optimal_f1_threshold = threshold_results.loc[threshold_results['f1'].idxmax(), 'threshold']
                summary['optimal_f1_threshold'] = optimal_f1_threshold
            
            # Save summary
            summary_file = os.path.join(self.results_dir, f'{model_name}_evaluation_summary.csv')
            summary.to_csv(summary_file, index=False)
            
            logger.info(f"Evaluation complete for {model_name}")
            logger.info(f"Summary saved to {summary_file}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}", exc_info=True)
            raise


def evaluate_model(model, X_test, y_test, X_train=None, y_train=None, 
                 model_name='appian_model', perform_cv=True):
    """
    Evaluate a trained model for Appian purchase prediction
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test targets
    X_train : pandas.DataFrame, optional
        Training features (for cross-validation)
    y_train : pandas.Series, optional
        Training targets (for cross-validation)
    model_name : str
        Name of the model
    perform_cv : bool
        Whether to perform cross-validation
        
    Returns:
    --------
    dict
        Dictionary of evaluation results
    """
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Perform evaluation
        results = evaluator.evaluate_and_save(
            model=model,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            perform_cv=perform_cv
        )
        
        logger.info(f"Model evaluation completed for {model_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}", exc_info=True)
        raise
