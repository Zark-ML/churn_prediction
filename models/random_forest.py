from models.abs_model import Model
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(Model):
    """
    Radnom Forest Model implementation of the abstract Model class.

    Methods:
        train(self, data, label): Train the Radnom Forest Classifier.
        predict(self, test): Make predictions using the trained model.
        save(self, path=None): Save the trained model to a file.
        load(self, path=None): Load a pre-trained model from a file.
        hyper_parameter(self, parameters_dict): Perform hyperparameter tuning.
        gs_parameter_tune(self, parameters): Perform grid search parameter tuning.
    """

    def __init__(self, name = 'RandomForestModel'):
        """
        Initializes a new RandomForestModel instance.

        Parameters:
            name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = RandomForestClassifier()
        self.hyper_parameters = None
