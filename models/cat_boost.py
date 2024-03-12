from models.abs_model import Model
from catboost import CatBoostClassifier


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
    def hyper_parameter(self, parameters_dict):
        """
        Abstract method for hyperparameter tuning.

        Parameters:
            parameters_dict (dict): Dictionary containing hyperparameters.
        """
        self.hyper_parameters = parameters_dict
        self.model = CatBoostClassifier(random_state=42, task_type = 'GPU')
        self.model.set_params(**self.hyper_parameters)