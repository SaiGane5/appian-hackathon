# weight_tuning.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define pipelines
    svm = ('svm', Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC(probability=True, random_state=42))
    ]))
    lgbm = ('lgbm', Pipeline([
        ('lgbm', LGBMClassifier(random_state=42))
    ]))
    estimators = [svm, lgbm]

    # Grid over w in [0,0.1,…,1.0]
    best = {'weights': None, 'roc': 0}
    for w1 in np.linspace(0, 1, 11):
        w2 = 1 - w1
        vc = VotingClassifier(estimators=estimators, voting='soft', weights=[w1, w2])
        vc.fit(X_train, y_train)
        roc = roc_auc_score(y_val, vc.predict_proba(X_val)[:,1])
        if roc > best['roc']:
            best = {'weights': [w1, w2], 'roc': roc}

    print(f"Best weights: {best['weights']}  Val ROC-AUC: {best['roc']:.4f}")

    # Retrain on full
    vc = VotingClassifier(estimators=estimators, voting='soft', weights=best['weights'])
    vc.fit(X, y)

    # Align & predict
    X_test_al = align_features(X.columns, X_test)
    preds = vc.predict(X_test_al)

    # Save model & submission
    os.makedirs('models', exist_ok=True)
    joblib.dump(vc, 'models/voting_weighted.pkl')

    sub = pd.DataFrame({'ID': ids, 'Target': preds.astype(int)})
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/blend_voting_weighted.csv', index=False)
    print("Saved submissions/blend_voting_weighted.csv")

if __name__ == '__main__':
    main()
