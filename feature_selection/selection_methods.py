import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import skfda
import mrmr
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import Lasso
from catboost import CatBoostClassifier

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
        return dict(list(self.feature_importance.items())[:n])


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
        model.fit(self.data, self.target, cat_features='auto')
        self.feature_importance = sorted(dict(zip(self.data.columns, model.get_feature_importance())).items(), key=lambda x: x[1], reverse=True)
