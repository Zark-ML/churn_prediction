from abs_model import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class DecisionTreeModel(Model):
    """
    Decision Tree Classifier implementation of the abstract Model class.

    Methods:
        train(self, data, label): Train the Decision Tree Classifier.
        predict(self, test): Make predictions using the trained model.
        save(self, path=None): Save the trained model to a file.
        load(self, path=None): Load a pre-trained model from a file.
        hyper_parameter(self, parameters_dict): Perform hyperparameter tuning.
        gs_parameter_tune(self, parameters): Perform grid search parameter tuning.
    """

    def __init__(self, name = 'DecisionTreeModel'):
        """
        Initializes a new DecisionTreeModel instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = DecisionTreeClassifier()
        self.hyper_parameters = None

    def train(self, data, label):
        """
        Train the Decision Tree Classifier.

        Parameters:
            data: The training data.
            label: The labels corresponding to the training data.
        """
        self.model.fit(data, label)
        self._is_trained = True
    
    
    def predict(self, test):
        """
        Make predictions using the trained model.

        Parameters:
            test: The data for making predictions.

        Returns:
            list: Predicted labels.
        """
        if not self._is_train():
            raise ValueError("Model is not trained.")
        return self.model.predict(test)


    def save(self, path=None):
        """
        Save the trained model to a file.

        Parameters:
            path (str): The path where the model should be saved.
        """
        if not self._is_train():
            raise ValueError("Model is not trained.")
        # Save the model using appropriate serialization method (e.g., joblib, pickle)
        # Example: joblib.dump(self.model, path)

    def load(self, path=None):
        """
        Load a pre-trained model from a file.

        Parameters:
            path (str): The path from which the model should be loaded.
        """
        # Load the model using appropriate deserialization method (e.g., joblib, pickle)
        # Example: self.model = joblib.load(path)
        # self.__is_trained = True
        pass


    def hyper_parameter(self, parameters_dict):
        """
        Perform hyperparameter tuning.

        Parameters:
            parameters_dict (dict): Dictionary containing hyperparameters.
        """
        # if not self.__is_train():
        #     raise ValueError("Model is not trained.")

        self.hyper_parameters = parameters_dict
        self.model.set_params(**self.hyper_parameters)

    def gs_parameter_tune(self, data, label, parameters, scoring ):
        # if not self._is_train():
        #     raise ValueError("Model is not trained.")

        # Perform grid search
        grid_search = GridSearchCV(self.model, parameters, cv=5, scoring=scoring)
        grid_search.fit(data, label)

        # Get the best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best Parameters: {best_params}")
        print(f"Validation Accuracy: {best_score}")

    def __str__(self):
        """
        Returns the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.name

    def _is_train(self):
        """
        Checks if the model is trained.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        if self._is_trained:
            return True
        return False

