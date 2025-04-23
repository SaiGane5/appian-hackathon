# permutation_search.py
import os
import itertools
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
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
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipes = {
        'svm': Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))]),
        'lgbm': Pipeline([('lgbm', LGBMClassifier(random_state=42))]),
        'xgb': Pipeline([('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
        'lr': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))])
    }

    best = {'combo': None, 'roc': 0}
    for k in range(2, len(pipes)+1):
        for combo in itertools.combinations(pipes, k):
            estimators = [(name, pipes[name]) for name in combo]
            vc = VotingClassifier(estimators=estimators, voting='soft')
            vc.fit(X_tr, y_tr)
            roc = roc_auc_score(y_va, vc.predict_proba(X_va)[:,1])
            print(f"{combo} → Val ROC-AUC: {roc:.4f}")
            if roc > best['roc']:
                best = {'combo': combo, 'roc': roc}

    print("Best combo:", best['combo'], "ROC-AUC:", best['roc'])
    # Retrain best on full
    estimators = [(n, pipes[n]) for n in best['combo']]
    vc = VotingClassifier(estimators=estimators, voting='soft')
    vc.fit(X, y)
    X_test_al = align_features(X.columns, X_test)
    preds = vc.predict(X_test_al)

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(vc, f"models/voting_{'_'.join(best['combo'])}.pkl")

    sub = pd.DataFrame({'ID': ids, 'Target': preds.astype(int)})
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/blend_best_combo.csv', index=False)
    print("Saved submissions/blend_best_combo.csv")

if __name__ == '__main__':
    main()
