from abc import ABC, abstractmethod
from pathlib import Path


class Model(ABC):
    """
    Abstract base class for machine learning models.

    Attributes:
        name (str): The name of the model.
        __is_trained (bool): Flag indicating whether the model is trained.
        model: The machine learning model.

    Methods:
        __init__(self, name: str): Initializes the Model instance.
        __is_train(self): Checks if the model is trained.
        train(self, data, label): Abstract method to train the model.
        predict(self, test): Abstract method to make predictions.
        save(self, path=None): Abstract method to save the model.
        load(self, path=None): Abstract method to load the model.
        hyper_parameter(self, parameters_dict): Abstract method for hyperparameter tuning.
        gs_parameter_tune(self, parameters): Abstract method for grid search parameter tuning.
        __str__(self): Returns the name of the model.
    """

    def __init__(self, name: str):
        """
        Initializes a new Model instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self.__is_trained = False
        self.model = None

    def __is_train(self):
        """
        Checks if the model is trained.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        if self.__is_trained:
            return True
        logger.warning(f'{self} is not trained yet')
        return False

    @abstractmethod
    def train(self, data, label):
        """
        Abstract method to train the model.

        Parameters:
            data: The training data.
            label: The labels corresponding to the training data.
        """
        pass

    @abstractmethod
    def predict(self, test):
        """
        Abstract method to make predictions.

        Parameters:
            test: The data for making predictions.
        """
        pass

    @abstractmethod
    def save(self, path=None):
        """
        Abstract method to save the model.

        Parameters:
            path (str): The path where the model should be saved.
        """
        pass

    @abstractmethod
    def load(self, path=None):
        """
        Abstract method to load the model.

        Parameters:
            path (str): The path from which the model should be loaded.
        """
        pass

    @abstractmethod
    def hyper_parameter(self, parameters_dict):
        """
        Abstract method for hyperparameter tuning.

        Parameters:
            parameters_dict (dict): Dictionary containing hyperparameters.
        """
        pass

    @abstractmethod
    def gs_parameter_tune(self, parameters):
        """
        Abstract method for grid search parameter tuning.

        Parameters:
            parameters: Parameters for grid search.
        """
        pass

    def __str__(self):
        """
        Returns the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.name
