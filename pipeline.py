import pandas as pd
import numpy as np
from preprocessing.process import process_data
from sklearn.model_selection import train_test_split
from metrics import Metrics
from feature_selection.select_features import SelectFeatures
from preprocessing.process import process_data
from models.adaBoost import AdaBoostModel
from models.cat_boost import CatBoostModel
from models.xg_boost import XGBoostModel
from models.gradient_boosting import GradientBoostingModel
from models.random_forest import RandomForestModel
from models.decision_tree import DecisionTreeModel
from feature_selection.selection_methods import *
from models.select_model import SelectModel
import json
import pickle
from sklearn.metrics import f1_score, roc_auc_score


class Pipeline:
    def __init__(self, data = None, data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv', selected_model = True, 
                    model_path = 'saved_models/the_best_model.pkl', predict_labels_save_path = 'data/predicted_labels.csv',
                    models_list=[
                                    AdaBoostModel,
                                    CatBoostModel,
                                    XGBoostModel,
                                    GradientBoostingModel,
                                    RandomForestModel,
                                    DecisionTreeModel
                                ] ,
                    selection_methods_list=[
                        MRMR,
                        Xgb_Selection,
                        GBM_Selection,
                        Rf_Selection,
                        Lasso_Selection,
                        Catboost_Selection,
                        RFE_Selection,
                        PCA_Selection,
                        Shap_Selection
                     ]) -> None:
        
        self.data = pd.read_csv(data_path) if data_path else data
        self.selected_model = selected_model
        self.model_path = model_path
        self.models_list = models_list
        self.selection_methods_list = selection_methods_list
        self.model = None
        self.predict_labels_save_path = predict_labels_save_path
        
    def select_model(self):
        self.model = SelectModel(data = self.selected_data, target = self.preprocessed_label, models_list = self.models_list)()

        return self.model


    def test_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))
        y_pred = self.model.predict(self.selected_data) 
        print(f'{self.model.__class__.__name__} Has been loaded successfully!')
        print(f'Predicted labels saved in the file {self.predict_labels_save_path}')
        pd.DataFrame(y_pred).to_csv(self.predict_labels_save_path, index=False)

        return y_pred

    def __call__(self):
        if not self.selected_model:
            self.preprocessed_data, self.preprocessed_label = process_data(data=self.data.iloc[:, :-1], label=self.data.iloc[:,-1], label_encoding=True, data_path=None, label_path=None)
            self.selected_data = SelectFeatures(selection_methods_list = self.selection_methods_list, data = self.preprocessed_data, target = self.preprocessed_label)()
            
            # self.selected_data = pd.read_csv('data/Telco-Customer-Churn-encoded-data-FE-Features-Selected.csv')
            # self.preprocessed_label = pd.read_csv('data/Telco-Customer-Churn-encoded-label.csv')

            return self.select_model()
            

        else:
            self.preprocessed_data = process_data(data=self.data, label_encoding=False, data_path=None, label_path=None)
            with open('selected_features.json', 'r') as f:
                selected_features = json.load(f)
            self.selected_data = self.preprocessed_data[selected_features]            
            
            return self.test_model()

            
if __name__ == '__main__':
    pipeline = Pipeline(data_path='data/WA_Fn-UseC_-Telco-Customer-Churn_train.csv',selected_model=False)
    pipeline()
    print('Pipeline has been executed successfully!')

    test = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn_test.csv')
    pipline = Pipeline(data = test.iloc[:,:-1], selected_model=True, data_path=None, model_path='saved_models/the_best_model.pkl')
    y_pred = pipline()
    print(y_pred)
    print('Pipeline has been executed successfully!')

    y_test = test.iloc[:,-1]
    y_test = np.where(y_test == 'Yes', 1, 0)
    print('f1:', Metrics(y_test, y_pred).f1_score())
    print('roc_auc:', Metrics(y_test, y_pred).roc_auc())
    print('skroc_auc:', roc_auc_score(y_test, y_pred))
    print('skf1:', f1_score(y_test, y_pred, average='weighted'))
    print('churn acuracy score:', y_pred.sum()/y_test.sum())
    print(Metrics(y_test, y_pred).confusion_matrix())
    Metrics(y_test, y_pred).save_metrics(filename='saved_models/best_model_metrics.json')
    Metrics(y_test, y_pred).plot_confusion_matrix()

