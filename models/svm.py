from models.abs_model import Model
from sklearn.svm import SVC 
import numpy as np

class SVMModel(Model):
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

    def __init__(self, name = 'SVMModel'):
        """
        Initializes a new GradientBoostingModel instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = SVC(random_state=42)
        self.hyper_parameters = None
        self.parameters = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
            'gamma': ['scale', 'auto'],  # Kernel coefficient
            'degree': [2, 3, 4],  # Degree of the polynomial kernel function (only for poly kernel)
            'coef0': [0.0, 0.1, 0.5],  # Independent term in kernel function (only for poly and sigmoid kernels)
            'shrinking': [True, False],  # Whether to use the shrinking heuristic
            'class_weight': [None, 'balanced']  # Weights associated with classes (only for SVC)
        }
        