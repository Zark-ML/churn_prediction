import numpy as np
import pandas as pd
from feature_selection.selection_methods import *
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import json

class SelectFeatures:
    def __init__(self, data = None, target = None, selection_methods_list=[
                    MRMR,
                    Xgb_Selection,
                    GBM_Selection,
                    Rf_Selection,
                    Lasso_Selection,
                    Catboost_Selection,
                    RFE_Selection,
                    PCA_Selection,
                    Shap_Selection
                ]):
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_csv('data/Telco-Customer-Churn-encoded-data-FE.csv')
        self.target = target if isinstance(target, pd.DataFrame) else pd.read_csv('data/Telco-Customer-Churn-encoded-label.csv')
        self.selection_methods_list = selection_methods_list
        self.importances = pd.DataFrame()
        self.selected_features_dict = dict()

    def get_importances(self, save=False, path='data/feature_importances.csv'):
        for method in self.selection_methods_list:
            method_name = method.__name__
            print(f'Method: {method_name}')
            model = method(self.data, target=self.target)
            model.fit()
            method_importances = model.get_importances()
            print(f'Feature importances:\n {method_importances}')
            importances_dict = pd.DataFrame(method_importances, index=[method_name])
            self.importances = pd.concat([self.importances, importances_dict])

        if save:
            self.importances.to_csv(path, index=True)

    def get_selected_features_count(self):
        for method in self.selection_methods_list:
            accuracies = []

            features = self.importances.T[method.__name__]
            features = features.sort_values(ascending=False)


            X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)

                
            model = method(self.data, self.target).used_model

            scores = [features.index[0]]

            for i in features.index:
                if i not in scores:
                    model.fit(X_train[[*scores,i]], y_train)
                    y_pred = model.predict(X_test[[*scores,i]])
                    score = f1_score(y_test, y_pred, average='weighted')
                    accuracies.append(score)
                    scores.append(i)

            print(scores, method)
            print(accuracies, 'acc')
            self.selected_features_dict[f'{method.__name__}'] = {'index': np.argmax(np.array(accuracies)), 
                                                                 'score': np.max(np.array(accuracies))}
    def get_selected_data(self, save=False, path='data/Telco-Customer-Churn-encoded-data-FE-Features-Selected.csv'):
        self.selected_algorithm = sorted(self.selected_features_dict.items(), key=lambda x:x[1]['score'], reverse=True)[0]
        selected_features = self.importances.T[self.selected_algorithm[0]].sort_values(ascending=False).index[:self.selected_algorithm[1]['index']+1]
        self.selected_data = self.data[selected_features]
        if save:
            with open('./selected_features.json', 'w') as f:
                json.dump([*selected_features], f,ensure_ascii=False, indent=4)
            self.selected_data.to_csv(path, index=False)
        return self.selected_data


    def __call__(self):
        self.get_importances(save=True)
        self.get_selected_features_count()
        self.get_selected_data(save=True)
        print('Selected features:', self.selected_data.columns)
        print('Selected algorithm:', self.selected_algorithm)
        print('Selected features count:', len(self.selected_data.columns))
        print('Selected features saved to:', 'data/Telco-Customer-Churn-encoded-data-FE-Features-Selected.csv')
        print('Selected columns saved to:', 'data/selected_features.json')
        print('Successfully selected features!')
        return self.selected_data

if __name__ == '__main__':
    SelectFeatures()()