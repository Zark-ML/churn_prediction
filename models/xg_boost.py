from models.abs_model import Model
import xgboost as xgb
import numpy as np

class XGBoostModel(Model):
    """
    XGBoost implementation of the abstract Model class.

    Methods:
        train(self, data, label): Train the XGBoost classifier.
        predict(self, test): Make predictions using the trained model.
        save(self, path=None): Save the trained model to a file.
        load(self, path=None): Load a pre-trained model from a file.
        hyper_parameter(self, parameters_dict): Perform hyperparameter tuning.
        gs_parameter_tune(self, parameters): Perform grid search parameter tuning.
    """

    def __init__(self, name = 'XGBoostModel'):
        """
        Initializes a new XGBoostModel instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = xgb.XGBClassifier(random_state=42)
        self.hyper_parameters = None
        self.parameters = {
            'max_depth': range(3, 10),  # Similar to DecisionTree
            'learning_rate': 10.0 ** np.arange(-5, 0),  # Equivalent to eta in XGBoost
            'n_estimators': range(50, 200, 50),  # Number of gradient boosted trees
            'subsample': np.arange(0.5, 1.0, 0.1),  # Subsample ratio of the training instances
            'colsample_bytree': np.arange(0.5, 1.0, 0.1)  # Subsample ratio of columns when constructing each tree
        }
