import os
import pandas as pd
import joblib
from data_loader import load_data
from feature_engineering import FeatureEngineer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def main(
    train_path: str = 'train.csv',
    test_path: str = 'test.csv',
    submission_path: str = 'submissions/blend_submission.csv',
    val_size: float = 0.2,
    random_state: int = 42
):
    # 1. Load & preprocess
    train_df = load_data(train_path, is_train=True)
    test_df  = load_data(test_path,  is_train=False)
    test_ids = test_df['ID'].copy()

    fe = FeatureEngineer(visualize=False)
    train_eng = fe.fit_transform(train_df)
    test_eng  = fe.transform(test_df)

    # 2. Prepare features & split for blending
    X = train_eng.drop(['ID', 'Target'], axis=1)
    y = train_eng['Target']
    X_test = test_eng.drop(['ID'], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    # 3. Base models pipelines
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC(probability=True, random_state=random_state))
    ])
    lgbm_pipe = Pipeline([
        ('lgbm', LGBMClassifier(random_state=random_state))
    ])

    # 4. Voting ensemble
    blender = VotingClassifier(
        estimators=[
            ('svm',  svm_pipe),
            ('lgbm', lgbm_pipe)
        ],
        voting='soft'
    )

    # 5. Train ensemble
    blender.fit(X_train, y_train)

    # 6. Validation evaluation
    y_val_pred  = blender.predict(X_val)
    y_val_proba = blender.predict_proba(X_val)[:, 1]
    print("=== Validation Classification Report ===")
    print(classification_report(y_val, y_val_pred))
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    # 7. Align test features with training schema
    train_cols = X_train.columns
    X_test_aligned = X_test.copy()
    # Drop any extra columns not seen in training
    extra_cols = set(X_test_aligned.columns) - set(train_cols)
    if extra_cols:
        X_test_aligned = X_test_aligned.drop(columns=list(extra_cols))
    # Add missing columns that were in training
    missing_cols = set(train_cols) - set(X_test_aligned.columns)
    for col in missing_cols:
        X_test_aligned[col] = 0
    # Reorder to match training
    X_test_aligned = X_test_aligned[train_cols]

    # 8. Final predictions on aligned test set
    y_test_pred = blender.predict(X_test_aligned)

    # 9. Save submission
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission = pd.DataFrame({
        'ID':     test_ids,
        'Target': y_test_pred.astype(int)
    })
    submission.to_csv(submission_path, index=False)
    print(f"Blended submission saved to → {submission_path}")

    # 10. Optionally save the blender for later
    os.makedirs('models', exist_ok=True)
    joblib.dump(blender, os.path.join('models', 'blend_svm_lgbm.pkl'))
    print("Blender model saved to models/blend_svm_lgbm.pkl")


if __name__ == '__main__':
    main()