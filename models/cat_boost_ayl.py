import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# Define objective function for Optuna
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_loguniform("random_strength", 1e-9, 10),
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 1, 10)
    }

    clf = CatBoostClassifier(**params, verbose=True, task_type='GPU')

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=True)
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
best_model = CatBoostClassifier(**best_params, verbose=True, task_type='GPU')
best_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)

# Evaluate on test set
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Best parameters found:", best_params)
print("ROC-AUC score on test set:", roc_auc)

import pickle

# Save the best model as a pickle file
with open('best_cat_boost.pkl', 'wb') as f:
    pickle.dump(best_model, f)
