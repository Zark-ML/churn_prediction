import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.adaBoost import AdaBoostModel
from models.cat_boost import CatBoostModel
from models.xg_boost import XGBoostModel
from models.gradient_boosting import GradientBoostingModel
from models.random_forest import RandomForestModel
from models.decision_tree import DecisionTreeModel
from metrics import Metrics

import json

class SelectModel:
    """
    Selects the best machine learning model for a given dataset.

    Args:
        data_path (str, optional): Path to the dataset CSV file. Defaults to '../data/Telco-Customer-Churn-encoded-data-FE-Features-Selected.csv'.
        target_path (str, optional): Path to the target labels CSV file. Defaults to '../data/Telco-Customer-Churn-encoded-label.csv'.
        models_list (list, optional): List of machine learning models to evaluate. Defaults to a predefined list of models.

    Attributes:
        data (pd.DataFrame): The input data.
        target (pd.Series): The target labels.
        models_list (list): List of machine learning models.
        best_model (object): The best model based on evaluation scores.
        best_score (float): The highest evaluation score achieved.
        X_train, X_test, y_train, y_test (pd.DataFrame): Train and test data splits.

    Methods:
        get_acuracies(): Evaluates each model and selects the best one.
    """


    def __init__(self, data_path = 'data/Telco-Customer-Churn-encoded-data-FE-Features-Selected.csv',
                 target_path = 'data/Telco-Customer-Churn-encoded-label.csv',
                 models_list=[
                    AdaBoostModel,
                    CatBoostModel,
                    XGBoostModel,
                    GradientBoostingModel,
                    RandomForestModel,
                    DecisionTreeModel
                ]):
        self.data = pd.read_csv(data_path)
        self.target = pd.read_csv(target_path)
        self.models_list = models_list
        self.best_model = None
        self.best_score = float('-inf')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
        with open(f'.saved_models/models_with_best_params.json', 'w') as f:
            json.dump({}, f)

    
    def get_acuracies(self):
        for model in self.models_list:
            current_model_name = model.__name__
            print(f'Method: {current_model_name}')
            current_model = model(f'{current_model_name}_model')
            current_model_best_params, current_model_resempling = current_model.gs_parameter_tune(self.data, self.target)
            current_model_score, current_model = self.get_model_score(model, current_model_best_params, current_model_resempling)
            if current_model_score > self.best_score:
                self.best_score = current_model_score
                self.best_model = current_model

    def get_model_score(self, model, best_params, resampling):
        model_with_best_params = model(f'{model.__name__}_model_with_best_params')
        model_with_best_params.hyper_parameter(best_params)
        if(resampling):
            model_with_best_params.fit_with_resampling(self.X_train, self.y_train)
        else:
            model_with_best_params.fit(self.X_train, self.y_train)

        y_pred = model_with_best_params.predict(self.X_test)
        score = Metrics(self.y_test, y_pred).f1_score()
        print(f'F1 Score for {model.__name__} with best params: {score}')
        with open(f'.saved_models/models_with_best_params.json', 'r') as f:
            models_with_best_params_dict = json.load(f)
        models_with_best_params_dict[model.__name__] = {'best_params': best_params, 'resampling': resampling, 'score': score}
        with open(f'.saved_models/models_with_best_params.json', 'w') as f:
            json.dump(models_with_best_params_dict, f)

        return (score, model_with_best_params)
        

    def __call__(self):
        print('Selecting the best model...')
        self.get_acuracies()
        print(f'Best model: {self.best_model.__name__},\n Best score: {self.best_score}')
        self.best_model.save(f'saved_models/{self.best_model.__name__}_model.pkl')
        return self.best_model

if __name__ == '__main__':
    SelectModel()()