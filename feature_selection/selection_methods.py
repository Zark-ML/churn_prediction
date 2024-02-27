import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mrmr
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

import pandas as pd

def get_non_numerical_columns(df):
    non_numerical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    return non_numerical_columns

class Feature_Selection:
    """A class for feature selection algorithms.

    This class provides methods for feature selection based on various algorithms.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: A pandas Series containing target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the Feature_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: A pandas Series containing target feature.
            method: A string specifying the feature selection method.
        """
    
        self.data = data
        self.target = target
        self.method = method
        self.feature_importance = dict()

    def get_importances(self, n=None):
        """Get the feature importances based on the specified method.

        Args:
            n: An integer specifying the number of top features to return.

        Returns:
            A dictionary containing the feature importances sorted in descending order.
        """

        if n is None:
            return self.feature_importance
        return dict(list(self.feature_importance)[:n])


class mrmr_selection(Feature_Selection):
    """A class for mRMR feature selection.

    This class provides methods for mRMR feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the mrmr_selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the mrmr_selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        self.feature_importance = mrmr.mRMR(self.data, self.target, self.method, n)


class xgb_selection(Feature_Selection):
    """A class for XGBoost feature selection.

    This class provides methods for XGBoost feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the xgb_selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the xgb_selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        xgb = XGBClassifier(objective='binary:logistic', random_state=42)
        xgb.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, xgb.feature_importances_))

class rf_selection(Feature_Selection):
    """A class for Random Forest feature selection.

    This class provides methods for Random Forest feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the rf_selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the rf_selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, rf.feature_importances_))

class lasso_selection(Feature_Selection):
    """A class for Lasso feature selection.

    This class provides methods for Lasso feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the lasso_selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the lasso_selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        lasso = Lasso(alpha=0.1)
        lasso.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, lasso.coef_))

class catboost_selection(Feature_Selection):
    """A class for CatBoost feature selection.

    This class provides methods for CatBoost feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the catboost_selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the catboost_selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        
        model = CatBoostClassifier(iterations=100,  
                               depth=6,        
                               learning_rate=0.1,
                               loss_function='Logloss',
                               verbose=False)
        model.fit(self.data, self.target, cat_features=get_non_numerical_columns(self.data))
        self.feature_importance = sorted(dict(zip(self.data.columns, model.get_feature_importance())).items(), key=lambda x: x[1], reverse=True)

class RFE_Selection(Feature_Selection):
    """A class for Recursive Feature Elimination (RFE) feature selection.

    This class provides methods for RFE feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the RFE_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

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
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the feature importances sorted in descending order.
    """

    def __init__(self, data, target, method):
        """Initialize the GBM_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the GBM_Selection class.

        Args:
            n: An integer specifying the number of top features to return.
        """

        gbm = GradientBoostingClassifier()
        gbm.fit(self.data, self.target)
        self.feature_importance = dict(zip(self.data.columns, gbm.feature_importances_))

class PCA_Selection(Feature_Selection):
    """A class for Principal Component Analysis (PCA) feature selection.

    This class provides methods for PCA feature selection.

    Attributes:
        data: A pandas DataFrame containing the input features.
        target: Column name of target feature.
        method: A string specifying the feature selection method.
        feature_importance: A dictionary containing the variance explained by each principal component.
    """

    def __init__(self, data, target, method):
        """Initialize the PCA_Selection class.

        Args:
            data: A pandas DataFrame containing the input features.
            target: Column name of target feature.
            method: A string specifying the feature selection method.
        """

        super().__init__(data, target, method)

    def fit(self, n=None):
        """Fit the PCA_Selection class.

        Args:
            n: An integer specifying the number of top principal components to return.
        """

        pca = PCA(n_components=n)
        pca.fit(self.data)
        self.feature_importance = pca.explained_variance_ratio_


if __name__ == '__main__':
    df = pd.read_csv('/home/spartak/Desktop/Telco_new/churn_prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    selector = catboost_selection(df.drop('Churn', axis=1), df['Churn'], 'MIQ')
    selector.fit()
    print(selector.get_importances())