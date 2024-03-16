from models.abs_model import Model
from catboost import CatBoostClassifier
import numpy as np

class CatBoostModel(Model):
    """
    CatBoost implementation of the abstract Model class.

    Methods:
        train(self, data, label): Train the CatBoost classifier.
        predict(self, test): Make predictions using the trained model.
        save(self, path=None): Save the trained model to a file.
        load(self, path=None): Load a pre-trained model from a file.
        hyper_parameter(self, parameters_dict): Perform hyperparameter tuning.
        gs_parameter_tune(self, parameters): Perform grid search parameter tuning.
    """

    def __init__(self, name = 'CatBoostModel'):
        """
        Initializes a new CatBoostModel instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = CatBoostClassifier(random_state=42, verbose=False)
        self.hyper_parameters = None
        self.parameters = {
            'iterations': range(50, 200, 50),  # Number of boosting iterations
            'learning_rate': 10.0 ** np.arange(-5, 0),  # Step size shrinkage used in updates during training
            'depth': range(3, 10),  # Depth of the trees
            'l2_leaf_reg': 10.0 ** np.arange(-5, 0),  # L2 regularization term on weights
            'border_count': [32, 64, 128],  # The number of splits for numerical features
            'loss_function': ['Logloss', 'CrossEntropy'],  # Loss function to be optimized
            'eval_metric': ['Logloss', 'AUC'],  # Metric used for validation data
            # 'cat_features': ['categorical_feature_index']  # Specify categorical features indices
        }

        self.optuna_parameters = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_loguniform("random_strength", 1e-9, 10),
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 1, 10)
    }

    def hyper_parameter(self, parameters_dict):
        """
        Abstract method for hyperparameter tuning.

        Parameters:
            parameters_dict (dict): Dictionary containing hyperparameters.
        """
        self.hyper_parameters = parameters_dict
        self.model = CatBoostClassifier(random_state=42, verbose=False)
        self.model.set_params(**self.hyper_parameters)
    
    def load(self, path=None):
        super().load(path)
        if not isinstance(self.model, CatBoostClassifier):
            raise ValueError(f"Catboost cannot load {type(self.mode)}")