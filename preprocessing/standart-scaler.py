from preprocessing import AbstractPreprocessor
import numpy as np
import json

class StandardScaler(AbstractPreprocessor):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        """
        Initializes the StandardScaler.
        """
        super().__init__()
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        """
        Compute the mean and standard deviation of data for later scaling.

        Args:
            data (array-like): The data used to compute mean and standard deviation.

        Returns:
            StandardScaler: The fitted StandardScaler object.
        """
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        return self

    def transform(self, data):
        """
        Scale the data to zero mean and unit variance based on the computed mean and standard deviation.

        Args:
            data (array-like): The data to be scaled.

        Returns:
            array-like: The scaled data.
        
        Raises:
            ValueError: If scaler has not been fitted yet.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean_) / self.std_

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Args:
            data (array-like): The data used to fit and transform.

        Returns:
            array-like: The transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def save(self, file_path):
        """
        Save the scaler parameters to a JSON file.

        Args:
            file_path (str): The file path to save the parameters.

        Raises:
            IOError: If an error occurs while writing to the file.
        """
        with open(file_path, 'a') as file:
            try:
                scaling_dict = json.loads(file_path)

            except:
                scaling_dict = dict()
            
            scaling_dict['Standard_Scaler'] = {'mean': self.mean_.tolist(), 'std': self.std_.tolist()}

            json.dump(scaling_dict, file)

    def load(self, file_path):
        """
        Load the scaler parameters from a JSON file.

        Args:
            file_path (str): The file path to load the parameters from.

        Raises:
            IOError: If the file does not exist or an error occurs while reading from it.
        """
        with open(file_path, 'r') as file:
            params = json.load(file)
            self.mean_ = np.array(params['mean'])
            self.std_ = np.array(params['std'])
