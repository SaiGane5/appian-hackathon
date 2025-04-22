# ===== model.py =====
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
import shap
import numpy as np

def select_features(model, X, top_n=30):
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = np.abs(shap_values[0]).mean(0)
    else:
        shap_values = np.abs(shap_values).mean(0)
    top_features = X.columns[np.argsort(shap_values)[-top_n:]]
    return top_features

def train_final_model(X, y):
    # Base model parameters
    lgbm_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }

    xgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'scale_pos_weight': 1.56,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }

    cat_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 0.1,
        'random_seed': 42,
        'auto_class_weights': 'Balanced',
        'verbose': 0
    }

    # Base models
    base_models = [
        ('lgbm', LGBMClassifier(**lgbm_params)),
        ('xgb', XGBClassifier(**xgb_params)),
        ('catboost', CatBoostClassifier(**cat_params))
    ]

    # Meta model with feature scaling
    meta_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', 
                          random_state=42, 
                          max_iter=1000,
                          C=0.1)
    )

    # Stacking classifier
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='predict_proba',
        n_jobs=-1
    )

    # Hyperparameter optimization using BayesSearchCV
    opt = BayesSearchCV(
        stack,
        {
            'final_estimator__logisticregression__C': (1e-3, 1e3, 'log-uniform'),
            'final_estimator__logisticregression__penalty': ['l2', None]  # Use None instead of 'none'
        },
        n_iter=15,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )

    # Fit the optimized model
    opt.fit(X, y)

    # Return the best model
    return opt.best_estimator_
    