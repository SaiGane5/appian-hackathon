import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Any, Union

# Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                            f1_score, confusion_matrix, classification_report,
                            average_precision_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.inspection import permutation_importance

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier, 
                             VotingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression

# Hyperparameter optimization
import optuna
from optuna.integration import OptunaSearchCV
from skopt import BayesSearchCV

# Feature importance
import shap

def select_features(X: pd.DataFrame, y: pd.Series, 
                   method: str = 'shap', 
                   model: Optional[Any] = None,
                   top_n: int = 30) -> List[str]:
    """
    Select the best features using various methods.
    
    Args:
        X: Feature dataframe
        y: Target series
        method: Feature selection method ('shap', 'permutation', 'rfecv')
        model: Pre-trained model (required for shap and permutation)
        top_n: Number of top features to select
        
    Returns:
        List of selected feature names
    """
    if method == 'shap' and model:
        # SHAP-based feature selection
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            # For multi-class, take the class 1 (positive class)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_names = X.columns.tolist()
        
        # Create dataframe with feature names and importance
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Select top features
        selected_features = shap_df.head(top_n)['Feature'].tolist()
        
    elif method == 'permutation' and model:
        # Permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        
        # Select top features
        selected_features = [X.columns[i] for i in sorted_idx[:top_n]]
        
    elif method == 'rfecv':
        # Recursive feature elimination with cross-validation
        if model is None:
            model = LGBMClassifier(random_state=42)
            
        selector = RFECV(
            estimator=model,
            step=1,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc',
            min_features_to_select=min(20, X.shape[1] // 2),
            n_jobs=-1
        )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.support_].tolist()
        
        # Limit to top_n if needed
        if len(selected_features) > top_n:
            # Use model's feature importances to rank the selected features
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_[selector.support_]
                top_indices = np.argsort(importances)[::-1][:top_n]
                selected_features = [selected_features[i] for i in top_indices]
            else:
                selected_features = selected_features[:top_n]
    else:
        # Default to all features
        selected_features = X.columns.tolist()[:top_n]
        
    return selected_features

def optimize_lgbm(X: pd.DataFrame, y: pd.Series, 
                 cv_folds: int = 5, 
                 n_trials: int = 50) -> Dict:
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        X: Feature dataframe
        y: Target series
        cv_folds: Number of cross-validation folds
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of optimized parameters
    """
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            LGBMClassifier(**param),
            X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores.mean()
    
    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['boosting_type'] = 'gbdt'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['class_weight'] = 'balanced'
    
    return best_params

def optimize_xgb(X: pd.DataFrame, y: pd.Series, 
                cv_folds: int = 5, 
                n_trials: int = 50) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X: Feature dataframe
        y: Target series
        cv_folds: Number of cross-validation folds
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of optimized parameters
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Calculate class weight
        pos_weight = (len(y) - sum(y)) / sum(y)
        param['scale_pos_weight'] = pos_weight
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            XGBClassifier(**param),
            X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores.mean()
    
    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters
    best_params = study.best_params
    pos_weight = (len(y) - sum(y)) / sum(y)
    best_params['scale_pos_weight'] = pos_weight
    best_params['eval_metric'] = 'auc'
    best_params['use_label_encoder'] = False
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    return best_params

def optimize_catboost(X: pd.DataFrame, y: pd.Series, 
                     cv_folds: int = 5, 
                     n_trials: int = 50) -> Dict:
    """
    Optimize CatBoost hyperparameters using Optuna.
    
    Args:
        X: Feature dataframe
        y: Target series
        cv_folds: Number of cross-validation folds
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of optimized parameters
    """
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_seed': 42,
            'eval_metric': 'AUC',
            'auto_class_weights': 'Balanced',
            'verbose': 0
        }
        
        if param['bootstrap_type'] == 'Bayesian':
            param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif param['bootstrap_type'] == 'Bernoulli':
            param['subsample'] = trial.suggest_float('subsample', 0.5, 1)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            CatBoostClassifier(**param),
            X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores.mean()
    
    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters
    best_params = study.best_params
    best_params['eval_metric'] = 'AUC'
    best_params['auto_class_weights'] = 'Balanced'
    best_params['random_seed'] = 42
    best_params['verbose'] = 0
    
    return best_params

def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, 
                        cv_folds: int = 5, 
                        use_smote: bool = True,
                        scoring: str = 'roc_auc') -> Tuple[float, float]:
    """
    Perform cross-validation of a model with optional SMOTE.
    
    Args:
        model: Model to evaluate
        X: Feature dataframe
        y: Target series
        cv_folds: Number of cross-validation folds
        use_smote: Whether to use SMOTE for handling class imbalance
        scoring: Scoring metric
        
    Returns:
        Tuple of mean score and standard deviation
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)
        else:
            X_train_res, y_train_res = X_train_fold, y_train_fold
        
        # Fit and predict
        model.fit(X_train_res, y_train_res)
        
        if scoring == 'roc_auc':
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_proba)
        elif scoring == 'average_precision':
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            score = average_precision_score(y_val_fold, y_proba)
        else:
            y_pred = model.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred)
        
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def find_optimal_threshold(model, X: pd.DataFrame, y: pd.Series, 
                          cv_folds: int = 5, 
                          metric: str = 'f1') -> float:
    """
    Find the optimal classification threshold using cross-validation.
    
    Args:
        model: Trained model
        X: Feature dataframe
        y: Target series
        cv_folds: Number of cross-validation folds
        metric: Metric to optimize ('f1', 'precision', 'recall', 'geometric_mean')
        
    Returns:
        Optimal threshold value
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    thresholds_list = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train on this fold
        model.fit(X_train_fold, y_train_fold)
        
        # Get probabilities
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val_fold, y_proba)
        
        # Calculate F1 score for each threshold
        f1_scores = []
        for i in range(len(thresholds)):
            if metric == 'f1':
                score = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
            elif metric == 'precision':
                score = precision[i]
            elif metric == 'recall':
                score = recall[i]
            elif metric == 'geometric_mean':
                score = np.sqrt(precision[i] * recall[i])
            f1_scores.append(score)
        
        # Find threshold with highest F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        thresholds_list.append(best_threshold)
    
    # Return the average of the best thresholds
    return np.mean(thresholds_list)

def train_final_model(X: pd.DataFrame, y: pd.Series, 
                     n_trials: int = 30, 
                     use_feature_selection: bool = True,
                     n_features: int = 50,
                     calibrate_threshold: bool = True) -> Tuple[Any, float, List[str]]:
    """
    Train an optimized ensemble model for customer purchase prediction.
    
    Args:
        X: Feature dataframe
        y: Target series
        n_trials: Number of hyperparameter optimization trials
        use_feature_selection: Whether to perform feature selection
        n_features: Number of features to select
        calibrate_threshold: Whether to find optimal threshold
        
    Returns:
        Tuple of (trained model, optimal threshold, selected features)
    """
    # Step 1: Optimize base models
    print("Optimizing LightGBM...")
    lgbm_params = optimize_lgbm(X, y, n_trials=n_trials)
    
    print("Optimizing XGBoost...")
    xgb_params = optimize_xgb(X, y, n_trials=n_trials)
    
    print("Optimizing CatBoost...")
    cat_params = optimize_catboost(X, y, n_trials=n_trials)
    
    # Step 2: Train optimized base models
    base_models = [
        ('lgbm', LGBMClassifier(**lgbm_params)),
        ('xgb', XGBClassifier(**xgb_params)),
        ('catboost', CatBoostClassifier(**cat_params))
    ]
    
    # Step 3: Feature selection (optional)
    selected_features = X.columns.tolist()
    if use_feature_selection:
        # Train temporary model for feature selection
        temp_model = LGBMClassifier(**lgbm_params)
        temp_model.fit(X, y)
        
        # Select top features
        selected_features = select_features(X, y, method='shap', model=temp_model, top_n=n_features)
        print(f"Selected {len(selected_features)} features")
        
        # Filter X to selected features only
        X_selected = X[selected_features]
    else:
        X_selected = X
    
    # Step 4: Create a meta-learner pipeline
    meta_learner = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=0.1, 
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # Step 5: Create a stacking ensemble
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        verbose=1
    )
    
    # Step 6: Make a voting ensemble that combines the stack with the base models
    voting_ensemble = VotingClassifier(
        estimators=[
            ('stack', stack),
            ('lgbm', LGBMClassifier(**lgbm_params)),
            ('xgb', XGBClassifier(**xgb_params))
        ],
        voting='soft',
        weights=[0.6, 0.2, 0.2]  # Give more weight to the stacked model
    )
    
    # Step 7: Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_selected, y)
    
    # Step 8: Train the final model
    print("Training final ensemble model...")
    voting_ensemble.fit(X_res, y_res)
    
    # Step 9: Find optimal threshold (optional)
    threshold = 0.5  # Default
    if calibrate_threshold:
        print("Finding optimal classification threshold...")
        threshold = find_optimal_threshold(voting_ensemble, X_selected, y, metric='f1')
        print(f"Optimal threshold: {threshold:.4f}")
    
    # Step 10: Return the trained model and threshold
    return voting_ensemble, threshold, selected_features

def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                  threshold: float = 0.5) -> Dict:
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model: Trained model
        X: Feature dataframe
        y: Target series
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': (y_pred == y).mean(),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
        'avg_precision': average_precision_score(y, y_proba),
        'confusion_matrix': confusion_matrix(y, y_pred),
        'classification_report': classification_report(y, y_pred, output_dict=True)
    }
    
    return metrics

def plot_feature_importance(model: Any, X: pd.DataFrame, top_n: int = 20) -> None:
    """
    Plot feature importance for the given model.
    
    Args:
        model: Trained model
        X: Feature dataframe
        top_n: Number of top features to display
    """
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
    # For ensemble models, try to get feature importances from the first estimator
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
        importances = model.estimators_[0].feature_importances_
    else:
        print("Feature importance not available for this model type")
        return
    
    # Create dataframe for visualization
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_shap_summary(model: Any, X: pd.DataFrame, top_n: int = 20) -> None:
    """
    Plot SHAP summary for the given model.
    
    Args:
        model: Trained model
        X: Feature dataframe
        top_n: Number of top features to display
    """
    try:
        # For ensemble models, use the first base estimator
        if hasattr(model, 'estimators_'):
            base_model = model.estimators_[0]
        elif hasattr(model, 'named_steps'):
            base_model = model
        else:
            base_model = model
            
        # Sample data for SHAP (for efficiency)
        X_sample = X.sample(min(1000, len(X)), random_state=42)
        
        # Create explainer and get SHAP values
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class, take the class 1 (positive class)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=top_n, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return