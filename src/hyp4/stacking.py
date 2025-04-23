# stacking.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from data_pipeline import load_and_engineer

def align_features(train_cols, X):
    X = X.copy()
    extras = set(X.columns) - set(train_cols)
    if extras:
        X.drop(columns=extras, inplace=True)
    for c in set(train_cols) - set(X.columns):
        X[c] = 0
    return X[train_cols]

def main():
    X, y, X_test, ids = load_and_engineer()
    # Define base pipelines
    base = [
        ('svm',  Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])),
        ('lgbm', Pipeline([('lgbm', LGBMClassifier(random_state=42))])),
        ('xgb',  Pipeline([('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])),
        ('lr',   Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))]))
    ]

    # Build stacking classifier
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(random_state=42),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1
    )

    # Fit & evaluate on full (via OOF)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, va in skf.split(X, y):
        stack.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = stack.predict_proba(X.iloc[va])[:,1]
    print("Stacker CV ROC-AUC:", roc_auc_score(y, oof))

    # Final fit & predict
    stack.fit(X, y)
    X_test_al = align_features(X.columns, X_test)
    preds = stack.predict(X_test_al)

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(stack, 'models/stacking.pkl')

    sub = pd.DataFrame({'ID': ids, 'Target': preds.astype(int)})
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/blend_stacking.csv', index=False)
    print("Saved submissions/blend_stacking.csv")

if __name__ == '__main__':
    main()
