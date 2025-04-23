from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from model_evaluation import ModelEvaluator
import json

def train_and_save_model(train_engineered, model_path='rf_model.pkl'):
    X = train_engineered.drop('Target', axis=1)
    y = train_engineered['Target']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    print("\nValidation Report:")
    print(classification_report(y_val, val_preds))
    # Save model
    import joblib
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    evaluator = ModelEvaluator(model, X_val, y_val)
    report = evaluator.generate_report()
    
    # Save feature schema
    with open('feature_columns.json', 'w') as f:
        json.dump(list(X.columns), f)
    return model, X.columns
