import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('train_model')

class ModelTrainer:
    """
    Comprehensive model training and evaluation for Appian purchase prediction.
    Supports multiple models, hyperparameter tuning, and evaluation metrics.
    """
    
    def __init__(self, model_dir='models', results_dir='results'):
        """
        Initialize the model trainer
        
        Parameters:
        -----------
        model_dir : str
            Directory to save trained models
        results_dir : str
            Directory to save training results
        """
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.debug_dir = 'debug'
        self.plots_dir = os.path.join(self.debug_dir, 'model_plots')
        
        # Create directories if they don't exist
        for directory in [self.model_dir, self.results_dir, self.debug_dir, self.plots_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.feature_columns = None
        
        logger.info("ModelTrainer initialized")
        
    def define_models(self):
        """Define all classification models to evaluate"""
        logger.info("Defining classification models")
        
        # Create dictionary of models to train
        models = {
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'random_forest': Pipeline([
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'gradient_boosting': Pipeline([
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            'xgboost': Pipeline([
                ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
            ]),
            'lightgbm': Pipeline([
                ('classifier', LGBMClassifier(random_state=42))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(random_state=42, probability=True))
            ]),
            'knn': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', KNeighborsClassifier())
            ]),
            'decision_tree': Pipeline([
                ('classifier', DecisionTreeClassifier(random_state=42))
            ])
        }
        
        logger.info(f"Defined {len(models)} classification models")
        return models
    
    def define_param_grids(self):
        """Define hyperparameter grids for each model type"""
        logger.info("Defining hyperparameter grids")
        
        param_grids = {
            'logistic_regression': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__class_weight': [None, 'balanced']
            },
            'random_forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
            },
            'gradient_boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__subsample': [0.8, 1.0]
            },
            'xgboost': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            },
            'lightgbm': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5, -1],
                'classifier__subsample': [0.8, 1.0],
                'classifier__class_weight': [None, 'balanced']
            },
            'svm': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            },
            'knn': {
                'classifier__n_neighbors': [3, 5, 7, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__p': [1, 2]  # Manhattan or Euclidean
            },
            'decision_tree': {
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__class_weight': [None, 'balanced']
            }
        }
        
        logger.info("Hyperparameter grids defined")
        return param_grids
    
    def prepare_data(self, X, y, test_size=0.2, stratify=True):
        """
        Prepare data for training by splitting into train/test sets
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature data
        y : pandas.Series
            Target variable
        test_size : float
            Proportion of data to use for testing
        stratify : bool
            Whether to stratify the split based on target
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data with test_size={test_size}, stratify={stratify}")
        
        # Save feature columns for later prediction
        self.feature_columns = X.columns.tolist()
        
        # Create train/test split
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")
        logger.info(f"Class distribution in train: {dict(pd.Series(y_train).value_counts())}")
        logger.info(f"Class distribution in test: {dict(pd.Series(y_test).value_counts())}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, cv=5, 
                                  scoring='roc_auc', find_best=True):
        """
        Train and evaluate all models
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        X_test : pandas.DataFrame
            Test features
        y_train : pandas.Series
            Training targets
        y_test : pandas.Series
            Test targets
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for model selection
        find_best : bool
            Whether to select the best model based on scoring
            
        Returns:
        --------
        dict
            Dictionary of trained models and their performance metrics
        """
        logger.info(f"Training and evaluating models with cv={cv}, scoring={scoring}")
        
        # Define models and parameter grids
        models = self.define_models()
        param_grids = self.define_param_grids()
        
        # Store results
        results = {}
        
        # Initialize best score for model selection
        best_score = 0
        best_model = None
        best_model_name = None
        
        # Train and evaluate each model
        for model_name, model in models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Setup cross-validation
                cv_stratified = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                
                # Create grid search with specified parameters
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    cv=cv_stratified,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                grid_search.fit(X_train, y_train)
                
                # Get best model and its parameters
                best_model_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_
                
                # Make predictions on test set
                y_pred = best_estimator.predict(X_test)
                y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
                
                # Calculate performance metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Store model and metrics
                results[model_name] = {
                    'model': best_estimator,
                    'best_params': best_model_params,
                    'cv_results': grid_search.cv_results_,
                    'best_cv_score': grid_search.best_score_,
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_roc_auc': roc_auc,
                    'confusion_matrix': cm,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Log results
                logger.info(f"{model_name} results:")
                logger.info(f"  Best parameters: {best_model_params}")
                logger.info(f"  Best CV {scoring}: {grid_search.best_score_:.4f}")
                logger.info(f"  Test accuracy: {accuracy:.4f}")
                logger.info(f"  Test precision: {precision:.4f}")
                logger.info(f"  Test recall: {recall:.4f}")
                logger.info(f"  Test F1: {f1:.4f}")
                logger.info(f"  Test ROC AUC: {roc_auc:.4f}")
                
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
                
                # Plot ROC curve
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(self.plots_dir, f'{model_name}_roc_curve.png'))
                plt.close()
                
                # Check if this is the best model
                if find_best and roc_auc > best_score:
                    best_score = roc_auc
                    best_model = best_estimator
                    best_model_name = model_name
                    logger.info(f"New best model: {model_name} with ROC AUC = {roc_auc:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}", exc_info=True)
                results[model_name] = {'error': str(e)}
        
        # Save the best model
        if find_best and best_model is not None:
            self.best_model = best_model
            self.best_model_name = best_model_name
            self.best_score = best_score
            
            logger.info(f"Best model: {best_model_name} with ROC AUC = {best_score:.4f}")
            
            # Save best model
            joblib.dump(best_model, os.path.join(self.model_dir, f'{best_model_name}_best.pkl'))
            
            # Save feature columns for inference
            with open(os.path.join(self.model_dir, 'feature_columns.json'), 'w') as f:
                json.dump(self.feature_columns, f)
        
        # Store all models
        self.models = results
        
        # Create comparison plot of all models
        self._plot_model_comparison(results)
        
        return results
    
    def _plot_model_comparison(self, results):
        """
        Plot comparison of model performances
        
        Parameters:
        -----------
        results : dict
            Dictionary of model results
        """
        logger.info("Creating model comparison plots")
        
        try:
            # Extract model names and metrics
            model_names = []
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            roc_auc_scores = []
            
            for model_name, result in results.items():
                if 'error' not in result:
                    model_names.append(model_name)
                    accuracy_scores.append(result['test_accuracy'])
                    precision_scores.append(result['test_precision'])
                    recall_scores.append(result['test_recall'])
                    f1_scores.append(result['test_f1'])
                    roc_auc_scores.append(result['test_roc_auc'])
            
            # Create dataframe for plotting
            metrics_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracy_scores,
                'Precision': precision_scores,
                'Recall': recall_scores,
                'F1': f1_scores,
                'ROC AUC': roc_auc_scores
            })
            
            # Save metrics to CSV
            metrics_df.to_csv(os.path.join(self.results_dir, 'model_metrics_comparison.csv'), index=False)
            
            # Plot bar charts for each metric
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
            
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                # Sort by metric score
                sorted_df = metrics_df.sort_values(metric, ascending=False)
                sns.barplot(x='Model', y=metric, data=sorted_df, palette='viridis')
                plt.title(f'Model Comparison - {metric}')
                plt.xlabel('Model')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                
                # Add value labels
                for i, v in enumerate(sorted_df[metric]):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
                    
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'model_comparison_{metric}.png'))
                plt.close()
                
            # Create radar chart for model comparison
            # Prepare data for radar chart
            metrics_df_radar = metrics_df.set_index('Model')
            
            # Plot radar chart
            plt.figure(figsize=(10, 8))
            
            # Number of variables
            categories = list(metrics_df_radar.columns)
            N = len(categories)
            
            # Create angles for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create subplot with polar projection
            ax = plt.subplot(111, polar=True)
            
            # Add lines for each model
            for model in metrics_df_radar.index:
                values = metrics_df_radar.loc[model].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Fix axis
            plt.xticks(angles[:-1], categories)
            ax.set_rlabel_position(0)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Model Comparison Radar Chart', size=15, y=1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'model_comparison_radar.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {e}", exc_info=True)
    
    def analyze_feature_importance(self, X):
        """
        Analyze feature importance from the best model
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature data with column names
        """
        logger.info("Analyzing feature importance")
        
        if self.best_model is None:
            logger.warning("No best model available for feature importance analysis")
            return
        
        try:
            # Get feature importance from the model
            feature_importance = None
            model_type = self.best_model_name
            
            # Extract the classifier from the pipeline
            if hasattr(self.best_model, 'named_steps') and 'classifier' in self.best_model.named_steps:
                classifier = self.best_model.named_steps['classifier']
                
                # Get feature importance based on model type
                if hasattr(classifier, 'feature_importances_'):
                    # Tree-based models
                    feature_importance = classifier.feature_importances_
                elif hasattr(classifier, 'coef_'):
                    # Linear models
                    feature_importance = np.abs(classifier.coef_[0])
                else:
                    logger.warning(f"Model {model_type} doesn't provide feature importance")
                    return
            else:
                logger.warning("Best model doesn't have a classifier step")
                return
            
            if feature_importance is not None:
                # Create DataFrame with feature names and importance scores
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                # Save to CSV
                importance_df.to_csv(os.path.join(self.results_dir, 'feature_importance.csv'), index=False)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
                plt.title(f'Top 20 Feature Importance - {model_type}')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'))
                plt.close()
                
                # Log top features
                logger.info(f"Top 10 important features: {importance_df.head(10)['Feature'].tolist()}")
                
                return importance_df
                
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}", exc_info=True)
    
    def save_all_models(self):
        """Save all trained models to disk"""
        logger.info("Saving all models")
        
        for model_name, result in self.models.items():
            if 'error' not in result and 'model' in result:
                try:
                    model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
                    joblib.dump(result['model'], model_path)
                    logger.info(f"Saved {model_name} to {model_path}")
                except Exception as e:
                    logger.error(f"Error saving {model_name}: {e}", exc_info=True)
    
    def train(self, X, y, test_size=0.2, cv=5, scoring='roc_auc', find_best=True, save_all=False):
        """
        Full training pipeline: prepare data, train models, analyze features
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature data
        y : pandas.Series
            Target variable
        test_size : float
            Proportion of data to use for testing
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for model selection
        find_best : bool
            Whether to select the best model based on scoring
        save_all : bool
            Whether to save all models or just the best one
            
        Returns:
        --------
        dict
            Dictionary of trained models and their performance metrics
        """
        logger.info("Starting full training pipeline")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size=test_size)
        
        # Train and evaluate models
        results = self.train_and_evaluate_models(
            X_train, X_test, y_train, y_test, 
            cv=cv, scoring=scoring, find_best=find_best
        )
        
        # Analyze feature importance
        if find_best:
            self.analyze_feature_importance(X)
        
        # Save all models if requested
        if save_all:
            self.save_all_models()
        
        logger.info("Training pipeline completed successfully")
        
        return results


def train_and_save_model(train_df, model_path='rf_model.pkl'):
    """
    Train and save a model for Appian purchase prediction
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data with features and target
    model_path : str
        Path to save the trained model
        
    Returns:
    --------
    tuple
        (trained_model, feature_columns)
    """
    try:
        logger.info("Starting model training")
        
        # Separate features and target
        if 'Target' not in train_df.columns:
            logger.error("Target column not found in training data")
            raise ValueError("Target column not found in training data")
            
        X = train_df.drop('Target', axis=1)
        y = train_df['Target']
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Train models
        results = trainer.train(X, y, test_size=0.2, cv=5, scoring='roc_auc')
        
        # Get best model
        best_model = trainer.best_model
        feature_columns = trainer.feature_columns
        
        # Save feature columns as JSON
        feature_columns_path = os.path.splitext(model_path)[0] + '_features.json'
        with open(feature_columns_path, 'w') as f:
            json.dump(feature_columns, f)
            
        logger.info(f"Model training completed. Best model: {trainer.best_model_name}")
        
        return best_model, feature_columns
        
    except Exception as e:
        logger.error(f"Error in train_and_save_model: {e}", exc_info=True)
        raise
