import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, Dict

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUC = make_scorer(roc_auc_score, needs_proba=True)
POS_WEIGHT = 0.61/0.39  # Class balance ratio

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        current_year = datetime.now().year
        ref_date = pd.to_datetime(f"{current_year}-12-31")
        
        # Ensure Dt_Customer is in datetime format
        X['Dt_Customer'] = pd.to_datetime(X['Dt_Customer'], errors='coerce', format='mixed')
        
        # Temporal features
        X['Age'] = current_year - X['Year_Birth']
        X['Tenure'] = (ref_date - X['Dt_Customer']).dt.days
        
        # Household composition
        X['TotalChildren'] = X['Kidhome'] + X['Teenhome']
        X['IncomePerMember'] = X['Income'] / (1 + X['TotalChildren'])
        
        # Spending analysis
        spend_cols = ['MntWines','MntFruits','MntMeatProducts',
                    'MntFishProducts','MntSweetProducts','MntGoldProds']
        purchase_cols = ['NumWebPurchases','NumCatalogPurchases',
                        'NumStorePurchases','NumDealsPurchases']
        
        X['Monetary'] = X[spend_cols].sum(axis=1)
        X['Frequency'] = X[purchase_cols].sum(axis=1)
        X['AvgSpendPerPurchase'] = X['Monetary'] / X['Frequency'].replace(0,1)
        
        # Spending proportions
        for col in spend_cols:
            X[f'{col}_Ratio'] = X[col] / X['Monetary'].replace(0,1)
        
        # Campaign response
        cmp_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']
        X['TotalCampaigns'] = X[cmp_cols].sum(axis=1)
        
        # Temporal cyclic features
        X['EnrollMonth'] = X['Dt_Customer'].dt.month
        X['EnrollMonth_sin'] = np.sin(2*np.pi*X['EnrollMonth']/12)
        X['EnrollMonth_cos'] = np.cos(2*np.pi*X['EnrollMonth']/12)
        
        # Interaction terms
        X['HighIncome_LowSpending'] = (X['Income'] > X['Income'].median()) & (X['Monetary'] < X['Monetary'].median())
        
        # Drop original columns
        drop_cols = ['ID','Year_Birth','Dt_Customer','EnrollMonth'] + spend_cols + purchase_cols + cmp_cols
        return X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

def build_pipeline(model, num_cols, cat_cols):
    """Build complete pipeline with preprocessing and model"""
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    
    return Pipeline([
        ('fe', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def get_models(num_cols, cat_cols):
    """Return base models with parameter grids"""
    return {
        'xgb': (XGBClassifier(use_label_encoder=False, random_state=42, 
                             scale_pos_weight=POS_WEIGHT),
               {'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [3, 5],
                'model__n_estimators': [200, 400],
                'model__subsample': [0.8, 1.0]}),
        
        'lgbm': (LGBMClassifier(random_state=42, class_weight='balanced'),
                {'model__num_leaves': [31, 63],
                 'model__learning_rate': [0.01, 0.1],
                 'model__n_estimators': [200, 400],
                 'model__reg_alpha': [0, 0.1]}),
        
        'cat': (CatBoostClassifier(random_state=42, silent=True,
                                  auto_class_weights='Balanced'),
               {'model__iterations': [200, 400],
                'model__depth': [4, 6],
                'model__learning_rate': [0.03, 0.1]}),
        
        'logit': (LogisticRegression(class_weight='balanced', max_iter=1000),
                 {'model__C': [0.1, 1, 10],
                  'model__solver': ['lbfgs', 'liblinear']})
    }

def run_pipeline(train_path: str, test_path: str, submission_path: str):
    # Load raw data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info(f"Data loaded: train {train.shape}, test {test.shape}")

    # Prepare data
    y = train['Target'].values
    sample = FeatureEngineer().fit_transform(train)
    num_cols = sample.select_dtypes(include=np.number).columns.drop('Target', errors='ignore')
    cat_cols = sample.select_dtypes(include='object').columns
    X = train.drop(columns=['Target'], errors='ignore')

    # Setup validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_models = []
    
    # Tune models
    for name, (model, params) in get_models(num_cols, cat_cols).items():
        logger.info(f"Tuning {name.upper()}...")
        pipe = build_pipeline(model, num_cols, cat_cols)
        search = RandomizedSearchCV(pipe, params, n_iter=15, cv=skf, 
                                   scoring=AUC, n_jobs=-1, random_state=42)
        search.fit(X, y)
        base_models.append((name, search.best_estimator_))
        logger.info(f"{name.upper()} Best AUC: {search.best_score_:.4f}")

    # Build stacking ensemble
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True),
        cv=skf, n_jobs=-1, passthrough=False
    )
    
    # Train and validate stack
    logger.info("Training stacking classifier...")
    stack.fit(X, y)
    cv_preds = cross_val_predict(stack, X, y, cv=skf, method='predict_proba')[:,1]
    logger.info(f"Stacking CV AUC: {roc_auc_score(y, cv_preds):.4f}")

    # Generate submission
    test_preds = stack.predict(test)
    submission = pd.read_csv(submission_path)
    submission['Target'] = test_preds
    submission.to_csv('submission.csv', index=False)
    logger.info("Submission saved")

if __name__ == '__main__':
    run_pipeline('../data/train.csv', '../data/test.csv', '../data/sample_submission.csv')