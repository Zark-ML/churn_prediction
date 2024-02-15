from abc import ABC, abstractmethod
from pathlib import Path
from helper import logger

class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.__is_trained = False
        self.model = None

    def __is_train(self):
        if self.__is_trained:
            return True
        logger.warning(f'{self} is not trained yet')
        return False

    @abstractmethod
    def train(self, data, label):
        pass

    @abstractmethod
    def predict(self, test):
        pass

    @abstractmethod
    def save(self, path=None):
        pass

    @abstractmethod
    def load(self, path=None):
        pass

    @abstractmethod
    def hyper_parameter(self, parameters_dict):
        pass

    @abstractmethod
    def gs_parameter_tune(self, parameters):
        pass


    def __str__(self):
        return self.name
