#!/usr/bin/env python
# refined_pipeline.py
"""
End-to-end refined ML pipeline for Appian purchase prediction:
- Robust data loading & preprocessing
- Advanced feature engineering
- Hyperparameter tuning via RandomizedSearchCV
- Cross-validated stacking and automated ensemble selection
- Final submission generation
"""
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Data loading & preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 1. Load Data

def load_data(path, is_train=True):
    df = pd.read_csv(path, parse_dates=['Dt_Customer'], na_values=['', 'Unknown', 'NA'])
    if is_train:
        return df
    else:
        return df

# 2. Feature Engineering

def engineer_features(df):
    df = df.copy()
    # Registration tenure
    today = pd.to_datetime('2025-04-23')
    df['Days_Since_Reg'] = (today - df['Dt_Customer']).dt.days
    df.drop('Dt_Customer', axis=1, inplace=True)
    # Age
    df['Age'] = today.year - df['Year_Birth']
    # Family
    df['FamilySize'] = 1 + df['Kidhome'] + df['Teenhome']
    df['ChildRatio'] = (df['Kidhome']+df['Teenhome'])/df['FamilySize']
    # Total spend & ratios
    prod_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    df['TotalSpend'] = df[prod_cols].sum(axis=1)
    for c in prod_cols:
        df[c+'_Pct'] = df[c]/df['TotalSpend'].replace(0,1)
    # Channel features
    ch_cols = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth']
    df['TotalPurchases'] = df[ch_cols[:3]].sum(axis=1)
    df['WebVisitRatio'] = df['NumWebVisitsMonth']/df['NumWebPurchases'].replace(0,1)
    # Campaign
    cmp_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']
    df['TotalAccepted'] = df[cmp_cols].sum(axis=1)
    # Drop unused
    df.drop(['ID','Year_Birth','Kidhome','Teenhome'], axis=1, inplace=True)
    return df

# 3. Build preprocess pipeline

def build_preprocessor(df):
    num_feats = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_feats = ['Education','Marital_Status']
    num_feats = [c for c in num_feats if c not in ['Target']]
    num_transform = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transform = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preproc = ColumnTransformer([
        ('num', num_transform, num_feats),
        ('cat', cat_transform, cat_feats)
    ])
    return preproc, num_feats, cat_feats

# 4. Model definitions & hyperparam grids

def get_models_and_params():
    models = {
        'lgbm': (LGBMClassifier(random_state=42), {
            'clf__n_estimators':[200,400], 'clf__learning_rate':[0.01,0.1], 'clf__max_depth':[-1,10,20]
        }),
        'xgb': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'clf__n_estimators':[200,400], 'clf__learning_rate':[0.01,0.1], 'clf__max_depth':[3,6]
        }),
        'svc': (SVC(probability=True, random_state=42), {
            'clf__C':[0.5,1,5], 'clf__kernel':['rbf'], 'clf__gamma':['scale','auto']
        }),
        'lr': (LogisticRegression(max_iter=1000, random_state=42), {
            'clf__C':[0.1,1,10], 'clf__penalty':['l2']
        })
    }
    return models

# 5. Training and evaluation

def train_base_models(X, y, preproc, models):
    trained = {}
    for name,(est,params) in models.items():
        pipe = Pipeline([('pre', preproc), ('clf', est)])
        search = RandomizedSearchCV(pipe, params, n_iter=10, scoring='roc_auc', cv=5, n_jobs=-1, random_state=42)
        search.fit(X,y)
        print(f"{name} best AUC: {search.best_score_:.4f}")
        trained[name] = search.best_estimator_
    return trained

# 6. Stacking ensemble

def build_stacker(trained):
    estimators = [(n,clf) for n,clf in trained.items()]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        n_jobs=-1
    )
    return stack

# 7. Main pipeline

def main():
    # paths
    train_path, test_path = 'train.csv','test.csv'
    # load
    df = load_data(train_path, True)
    df = engineer_features(df)
    X = df.drop('Target', axis=1)
    y = df['Target']
    # preprocess
    preproc, num_feats, cat_feats = build_preprocessor(df)
    # get models
    models = get_models_and_params()
    # train base
    trained = train_base_models(X,y,preproc,models)
    # stack
    stack = build_stacker(trained)
    stack.fit(X,y)
    # evaluate on train via CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X,y):
        stack.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = stack.predict_proba(X.iloc[val_idx])[:,1]
    print(f"Stacker CV ROC-AUC: {roc_auc_score(y,oof):.4f}")
    # final test
    test = load_data(test_path, False)
    test = engineer_features(test)
    X_test = test
    preds = stack.predict(X_test)
    # submission
    sub = pd.DataFrame({'ID': pd.read_csv(test_path)['ID'], 'Target': preds.astype(int)})
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/refined_submission.csv', index=False)
    print("Saved refined_submission.csv")
    # save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(stack, 'models/refined_stacker.pkl')

if __name__=='__main__':
    main()