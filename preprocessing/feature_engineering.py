from preprocessing.preprocessing import AbstractPreprocessor
import numpy as np
import pandas as pd

class FeatureEngineering():
    def __init__(self):
        super().__init__()
        self.data_ = None
        self.data_path = None

    def to_categoric(self,data = None, data_path = None):
        if data_path:
            self.data_path = data_path
            self.load(data_path)
        elif data:
            self.data_ = data
        self.data_.loc[self.data_['tenure'] == 0, 'tenure'] = 1
        self.data_['tenure_group'] = (self.data_['tenure']-1) // 12
        # labels = []
        # for i in range(1, 73, 12):
        #     labels.append("{0} - {1}".format(i, i + 11))
        # print(labels)
        # self.data_['tenure_group'] = pd.cut(self.data_['tenure'], bins=range(1,74,12), right=False, labels=labels)

        return self


    def process(self,data = None, data_path = None):
        if data_path:
            self.data_path = data_path
            self.load(data_path)
        elif data:
            self.data_ = data

        self.data_['Monthly/Total_Charges'] = (self.data_['MonthlyCharges'] / self.data_['TotalCharges'])
        self.data_['TotalCharges/tenure'] = (self.data_['TotalCharges'] / self.data_['tenure'])

        m_t_max = (self.data_['Monthly/Total_Charges'][self.data_['Monthly/Total_Charges'] != float('inf')]).max()

        self.data_['Monthly/Total_Charges'].replace(float('inf'),m_t_max, inplace=True)

        return self

    def load(self, data_path):
        """
        Load data from a given path.

        Args:
            data_path (str): The path to the input data file.
        """
        self.data_ = pd.read_csv(data_path)


    def save(self, data_path='../data/Telco-Customer-Churn-encoded-data-FE.csv'):
        """
        Save the cleaned data to a file.

        Args:
            data_path (str, optional): The path to save the cleaned data.
        """
        self.data_.to_csv(data_path, index=False)


if __name__ == '__main__':
    df = FeatureEngineering()\
    .to_categoric(data_path='../data/Telco-Customer-Churn-encoded-data.csv')\
    .process()\
    .save(data_path='../data/Telco-Customer-Churn-encoded-data-FE.csv')