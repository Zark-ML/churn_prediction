from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from metrics import cross_validate_with_resampling
from imblearn.over_sampling import SMOTE
import itertools
import random
import pickle
from tqdm import tqdm


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
    
    @abstractmethod
    def __init__(self, name: str):
        """
        Initializes a new Model instance.

        Parameters:
            name (str): The name of the model.
        """
        pass

    # @abstractmethod
    def _is_train(self):
        """
        Checks if the model is trained.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        if self._is_trained:
            return True
        return False


    # @abstractmethod
    def fit(self, data, label):
        """
        Train the Decision Tree Classifier.

        Parameters:
            data: The training data.
            label: The labels corresponding to the training data.
        """
        self.model.fit(data, label)
        self._is_trained = True

    # @abstractmethod
    def fit_with_resampling(self, data, label):
        """
        Train the Decision Tree Classifier.

        Parameters:
            data: The training data.
            label: The labels corresponding to the training data.
        """
        sm = SMOTE(sampling_strategy="all", random_state=42)
        Xr_train, yr_train = sm.fit_resample(data, label)
        self.model.fit(Xr_train, yr_train)
        self._is_trained = True


    # @abstractmethod
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

    # @abstractmethod
    def save(self, path=None):
        """
        Save the trained model to a file.

        Parameters:
            path (str): The path where the model should be saved.
        """
        if not self._is_train():
            raise ValueError("Model is not trained.")
        pickle.dump(self.model, open(path, 'wb'))

    # @abstractmethod
    def load(self, path=None):
        """
        Abstract method to load the model.

        Parameters:
            path (str): The path from which the model should be loaded.
        """
        self.model = pickle.load(open(path, 'rb'))
        self._is_trained = True

    # @abstractmethod
    def hyper_parameter(self, parameters_dict):
        """
        Abstract method for hyperparameter tuning.

        Parameters:
            parameters_dict (dict): Dictionary containing hyperparameters.
        """
        self.hyper_parameters = parameters_dict
        self.model.set_params(**self.hyper_parameters)

    # @abstractmethod
    def gs_parameter_tune(self, data, label, max_search=100):
    
        # Perform grid search
        parameters_list = [
            {parameter: value for parameter, value in zip(self.parameters.keys(), values)}
            for values in itertools.product(*self.parameters.values())
        ]
        #shuffle and take first max_search
        random.shuffle(parameters_list)
        parameters_list = parameters_list[:max_search]

        best_score = float('-inf')

        for params in tqdm(parameters_list):
            self.hyper_parameter(params)
            orig_score, resamppled_score = cross_validate_with_resampling(self, data, label, n_splits=5, random_state=42)
            resampling = (resamppled_score > orig_score)
            score = resamppled_score if resampling else orig_score
            if score > best_score:
                best_score = score
                best_params = params
                best_resampling = resampling
                # best_params['resampling'] = resampling

        print(f'Model: {self.name}')
        print(f"Best Parameters: {best_params}")
        print(f"Resempling: {best_resampling}")
        print(f"Validation Accuracy: {best_score}")
        return (best_params, best_resampling)

    # @abstractmethod
    def __str__(self):
        """
        Returns the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.name


# import itertools
# import random
# import optuna
# from tqdm import tqdm


#     def optuna_parameter_tune(self, data, label, parameters, max_search=100):
#         def objective(trial):
#             params = {name: trial.suggest_categorical(name, values) for name, values in parameters.items()}
#             self.hyper_parameter(params)
#             orig_score, resampled_score = cross_validate_with_resampling(self, data, label, n_splits=5, random_state=42)
#             resampling = (resampled_score > orig_score)
#             score = resampled_score if resampling else orig_score
#             return score
        
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=max_search)
        
#         best_params = study.best_params
#         best_resampling = (study.best_value > orig_score)
#         best_score = study.best_value

#         print(f'Model: {self.name}')
#         print(f"Best Parameters: {best_params}")
#         print(f"Resampling: {best_resampling}")
#         print(f"Validation Accuracy: {best_score}")
        
#         return best_params, best_resampling

# # Usage
# your_object = YourClass()
# your_object.gs_parameter_tune(data, label, parameters)
