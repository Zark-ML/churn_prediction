from models.abs_model import Model
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

class AdaBoostModel(Model):
    """
    Ada Boost Model implementation of the abstract Model class.

    Methods:
        train(self, data, label): Train the Ada Boost Model Classifier.
        predict(self, test): Make predictions using the trained model.
        save(self, path=None): Save the trained model to a file.
        load(self, path=None): Load a pre-trained model from a file.
        hyper_parameter(self, parameters_dict): Perform hyperparameter tuning.
        gs_parameter_tune(self, parameters): Perform grid search parameter tuning.
    """

    def __init__(self, name = 'AdaBoostModel'):
        """
        Initializes a new GradientBoostingModel instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = AdaBoostClassifier(random_state=42)
        self.hyper_parameters = None
        self.parameters = {
            'n_estimators': range(50, 200, 50),  # Number of weak learners to train iteratively
            'learning_rate': 10.0 ** np.arange(-5, 0),  # Weight applied to each classifier at each boosting iteration
            'algorithm': ['SAMME', 'SAMME.R']  # Algorithm to use
        }
        