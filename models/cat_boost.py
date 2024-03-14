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
        self.model = CatBoostClassifier(random_state=42,  task_type = 'GPU')
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

    def hyper_parameter(self, parameters_dict):
        """
        Abstract method for hyperparameter tuning.

        Parameters:
            parameters_dict (dict): Dictionary containing hyperparameters.
        """
        self.hyper_parameters = parameters_dict
        self.model = CatBoostClassifier(random_state=42, task_type = 'GPU')
        self.model.set_params(**self.hyper_parameters)