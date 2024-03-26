from preprocessing import AbstractPreprocessor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


class MyOneHotEncoder(AbstractPreprocessor):
    """
    One-hot encoding preprocessor class.

    Attributes:
        data_ (pd.DataFrame): The input data.
        data_path (str): The path to the input data file.
        encoded (pd.DataFrame): The encoded data after transformation.
    """

    def __init__(self):
        super().__init__()
        self.data_ = None
        self.data_path = None
        self.encoded = None
        self.label_ = None

    def fit(self, data_path=None, data=None):
        """
        Fit the preprocessor with data.

        Args:
            data_path (str, optional): The path to the input data file.
            data (pd.DataFrame, optional): The input data.

        Returns:
            OneHotencoder: The fitted preprocessor instance.
        """
        if data_path:
            self.data_path = data_path
            self.data_ = self.load(data_path)
        elif data.shape:
            self.data_ = data
        return self

    def transform(self):
        """
        Transform the input data using one-hot encoding.

        Raises:
            ValueError: If input data is not provided.

        Returns:
            pd.DataFrame: The encoded data.
        """
        if self.data_ is None:
            raise ValueError("Input data is not provided.")
        self.encoded = self.data_.copy()



        new_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                   'PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                   'PaymentMethod']

        enc = OneHotEncoder(categories='auto', drop="if_binary").fit(self.encoded[new_col])
        self.encoded = pd.get_dummies(self.encoded[new_col], columns=new_col)

        self.encoded = pd.concat([self.encoded, self.data_], axis=1)
        self.encoded = self.encoded.drop(columns=new_col)
        for i in self.encoded.columns[:-3]:
            self.encoded[i] = np.where(self.encoded[i] == True, 1, 0)

        drop_col = ['gender_Female', 'SeniorCitizen_0', 'Partner_No', 'Dependents_No', 'PhoneService_No',
                    'PaperlessBilling_No']
        

        self.encoded = self.encoded.drop(columns=drop_col)
        return self.encoded

    def fit_transform(self, data_path=None, data=None):
        """
        Fit and transform the input data.

        Args:
            data_path (str, optional): The path to the input data file.
            data (pd.DataFrame, optional): The input data.

        Returns:
            pd.DataFrame: The encoded data.
        """
        self.fit(data_path=data_path, data=data)
        return self.transform()
    def label_encoding(self, label_path, label):
        self.label_ = self.load(label_path) if label_path else label
        self.label_ = np.where(self.label_ == 'Yes', 1, 0)
        return self.label_

    def load(self, data_path):
        """
        Load data from a given path.

        Args:
            data_path (str): The path to the input data file.
        """
        tmp = pd.read_csv(data_path)
        return tmp

    def save(self, data_path='data/Telco-Customer-Churn-encoded-data.csv', label_path='data/Telco-Customer-Churn-encoded-label.csv'):
        """
        Save the encoded data to a file.

        Args:
            data_path (str, optional): The path to save the encoded data.
        """
        self.encoded.to_csv(data_path, index=False)
        if(not self.label_ is None):
            df = pd.DataFrame()
            df['Churn'] = self.label_
            df.to_csv(label_path, index=False)

        print(f'Encoded Data saved in path: {data_path}')
        print(f'Encoded Labels saved in path: {label_path}')


if __name__ == '__main__':
    from manipulations import Manipulations
    manipulation = Manipulations()
    data = manipulation.manipulate(data_path='../data/Telco-Customer-Churn-data.csv')
    # print(data)
    encode = MyOneHotEncoder()
    label = pd.read_csv('../data/Telco-Customer-Churn-label.csv')
    encode.fit(data = data, label = label)
    encode.transform()
    encode.save()
