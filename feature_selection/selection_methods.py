import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from mrmr import mrmr_classif
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from typing import Dict


def get_non_numerical_columns(df):
    non_numerical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    return non_numerical_columns

def non_numerical_to_category(df):
    non_numerical_columns = get_non_numerical_columns(df)
    for col in non_numerical_columns:
        df[col] = df[col].astype('category')
    return df

class Feature_Selection:
    """A class for feature selection algorithms.

    This class provides methods for feature selection based on various algorithms.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: A pandas Series containing target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the Feature_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: A pandas Series containing target feature.
        """
    
        self.data = data
        self.target = target
        self.feature_importance = dict()

    def get_importances(self, n=None):
        """Get the feature importances based on the specified method.

        Args:
            n: An integer specifying the number of top features to return.

        Returns:
            A dictionary containing the feature importances sorted in descending order.
        """

        if n is None:
            try:
                return dict(sorted(self.feature_importance.items(), key=lambda item: item[1], reverse=True))
            except:
                return self.feature_importance
        return sorted(dict(list(self.feature_importance)[:n]), key=lambda item: item[1], reverse=True)


class MRMR(Feature_Selection):
    def __init__(self, data, target, k=None, s=3):
        """
        :param k: number of features to select
        :param s: number of iterations
        """
        
        super().__init__(data, target)
        
        self.k = k
        self.s = s

    def fit(self):
        """
        :param data: pd.DataFrame, shape (n_samples, n_features)
        :param target: pd.DataFrame, shape (n_samples, n_labels)
        :return: dict, feature importance
        """
        if self.k is None:
            self.k = len(self.data.columns)
        rate_dict = {}
        for _ in range(self.s):
            chosen_idx = np.random.choice(len(self.data), replace=False, size=len(self.data) // self.s)
            data_chosen = self.data.iloc[chosen_idx].reset_index(drop=True)
            label_chosen = self.target.iloc[chosen_idx].reset_index(drop=True)
            selected_features = mrmr_classif(X=data_chosen, y=label_chosen, K=self.k, return_scores=True)
            F = selected_features[1]
            corr = selected_features[2]
            selected = []
            not_selected = list(data_chosen.columns)
            for _ in range(self.k):
                score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis=1).fillna(.00001)
                best = score.index[score.argmax()]
                if best in rate_dict:
                    rate_dict[best].append(score.max())
                else:
                    rate_dict[best] = [score.max()]
                selected.append(best)
                not_selected.remove(best)
        rate_dict_mean = {key:sum(value) / len(value) for key, value in rate_dict.items()}
        self.feature_importance = rate_dict_mean



class Xgb_Selection(Feature_Selection):
    """A class for XGBoost feature selection.

    This class provides methods for XGBoost feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the Xgb_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the Xgb_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        xgb = XGBClassifier(objective='binary:logistic', random_state=42, enable_categorical=True)
        xgb.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, xgb.feature_importances_))

class Rf_Selection(Feature_Selection):
    """A class for Random Forest feature selection.

    This class provides methods for Random Forest feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the Rf_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the Rf_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, rf.feature_importances_))

class Lasso_Selection(Feature_Selection):
    """A class for Lasso feature selection.

    This class provides methods for Lasso feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the Lasso_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the Lasso_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        lasso = Lasso(alpha=0.1)
        lasso.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, lasso.coef_))
        

class Catboost_Selection(Feature_Selection):
    """A class for CatBoost feature selection.

    This class provides methods for CatBoost feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the Catboost_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the Catboost_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """
        model = CatBoostClassifier(random_state=42, verbose=False)
        model.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, model.feature_importances_))
        
        
class RFE_Selection(Feature_Selection):
    """A class for Recursive Feature Elimination (RFE) feature selection.

    This class provides methods for RFE feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the RFE_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the RFE_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=n)
        selector = selector.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, selector.ranking_))

class GBM_Selection(Feature_Selection):
    """A class for Gradient Boosting Machine (GBM) feature selection.

    This class provides methods for GBM feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the GBM_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the GBM_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        gbm = GradientBoostingClassifier()
        gbm.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, gbm.feature_importances_))

# FIXME:
# TODO: PCA Selection should be wether removed or changed
class PCA_Selection(Feature_Selection):
    """A class for Principal Component Analysis (PCA) feature selection.

    This class provides methods for PCA feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the variance explained by each principal component.
    """

    def __init__(self, data, target):
        """Initialize the PCA_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the PCA_Selection class.

        Args:
            n: An integer specifying the number of top principal components to return.
        """

        pca = PCA(n_components=n)
        pca.fit(self.data)
        self.feature_importance = dict(zip(self.data.columns, pca.explained_variance_ratio_))

class Shap_Selection(Feature_Selection):
    """A class for Shap feature selection.

    This class provides methods for Shap feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target):
        """Initialize the Shap_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
        """

        super().__init__(data, target)

    def fit(self, n=None):
        """Fit the Shap_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """
        model = XGBClassifier(objective='binary:logistic', random_state=42, enable_categorical=True)
        self.data = self.data.replace([np.inf, -np.inf], 0)
        self.data = non_numerical_to_category(self.data)
        model.fit(self.data, self.target)
        explainer = shap.TreeExplainer(model=model)
        shap_values = explainer.shap_values(self.data)
        print(shap_values)
        self.feature_importance = dict(sorted(dict(zip(self.data.columns, np.abs(shap_values).mean(0))).items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    # Spartak you can use encoded dataset using path: "data/WA_Fn-UseC_-Telco-Customer-Churn-encoded.csv"
    df = pd.read_csv('/home/spartak/Desktop/Telco_new/churn_prediction/data/encoded.csv')
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    selector = Catboost_Selection(df.drop('Churn', axis=1), df['Churn'])
    selector.fit()
    sum = 0
    print(selector.get_importances())
    for i in selector.get_importances().values():
        sum += i
    
    print(sum)

    # df = pd.read_csv('/home/spartak/Desktop/Telco_new/churn_prediction/data/encoded.csv')
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(df.max(), inplace=True)

    # selector = MRMR(df, 5, k=None, s=3)
    # selector.fit(target=df['Churn'])
    # print(selector.get_importances())

    # sum = 0
    # for i in selector.get_importances().values():
    #     sum += i
    # print(sum)

    # df = pd.read_csv('/home/spartak/Desktop/Telco_new/churn_prediction/data/WA_Fn-UseC_-Telco-Customer-Churn_tenur_categorical.csv')
    # df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    # selector = Shap_Selection(df.drop('Churn', axis=1), df['Churn'])
    # selector.fit()
    # sum = 0
    # for i in selector.get_importances().values():
    #     sum += i
    # print(selector.get_importances())
    # print(sum)
