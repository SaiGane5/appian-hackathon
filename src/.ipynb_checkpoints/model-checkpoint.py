import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier


def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
    }

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(**param, random_state=42, class_weight='balanced'))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    return scores.mean()


def tune_hyperparameters(X, y, n_trials: int = 50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    return study.best_params_


def train_final_model(X, y, best_params: dict):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(**best_params, random_state=42, class_weight='balanced'))
    ])
    pipeline.fit(X, y)
    return pipeline