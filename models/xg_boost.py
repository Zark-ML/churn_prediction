import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Sample DataFrame, replace with your own

# Define objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-5, 1),
        "gamma": trial.suggest_loguniform("gamma", 1e-9, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 10),
    }

    clf = XGBClassifier(**params, verbosity=1, n_jobs=-1, device="cuda")

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)
        y_pred = clf.predict_proba(X_val)[:, 1]
        roc_auc_scores.append(roc_auc_score(y_val, y_pred))

    return sum(roc_auc_scores) / len(roc_auc_scores)

# Extract features and target
X = pd.read_csv('/home/spartak/Desktop/Telco_new/churn_prediction/data/Telco-Customer-Churn-encoded-data_Features-Selected.csv')
y = pd.read_csv('/home/spartak/Desktop/Telco_new/churn_prediction/data/Telco-Customer-Churn-encoded-label.csv')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Get best parameters
best_params = study.best_params

# Train final model
best_model = XGBClassifier(**best_params, verbosity=1, n_jobs=-1)
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

# Evaluate on test set
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Best parameters found:", best_params)
print("ROC-AUC score on test set:", roc_auc)


import pickle

# Save the best model as a pickle file
with open('best_model_xg_boost.pkl', 'wb') as f:
    pickle.dump(best_model, f)
