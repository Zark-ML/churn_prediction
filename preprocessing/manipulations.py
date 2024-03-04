from preprocessing.preprocessing import AbstractPreprocessor
import numpy as np
import pandas as pd

class Manipulations():
    def __init__(self):
        super().__init__()
        self.data_ = None
        self.data_path = None

    
    def manipulate(self,data = None, data_path = None):
        
        if data_path:
            self.data_path = data_path
            self.load(data_path)
        elif data:
            self.data_ = data

        self.data_ = self.data_.drop('customerID', axis=1)

        self.data_['TotalCharges'] = self.data_['TotalCharges'].replace(' ', 0)
        self.data_['TotalCharges'] = self.data_['TotalCharges'].astype('float64')

        return self.data_

    def load(self, data_path):
        """
        Load data from a given path.

        Args:
            data_path (str): The path to the input data file.
        """
        self.data_ = pd.read_csv(data_path)


    def save(self, data_path='data/cleaned.csv'):
        """
        Save the cleaned data to a file.

        Args:
            data_path (str, optional): The path to save the cleaned data.
        """
        self.data_.to_csv(data_path, index=False)

