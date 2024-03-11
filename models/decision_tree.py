from models.abs_model import Model
from sklearn.tree import DecisionTreeClassifier
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
        self.model = DecisionTreeClassifier(random_state=42)
        self.hyper_parameters = None
