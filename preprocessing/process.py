import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.manipulations import Manipulations
from preprocessing.feature_engineering import FeatureEngineering
from preprocessing.onehotencoder import MyOneHotEncoder

def process_data(data_path = 'data/Telco-Customer-Churn-data.csv',data = None, label = None, label_path = 'data/Telco-Customer-Churn-label.csv', label_encoding = False):
    data = pd.read_csv(data_path) if data_path else data
    label = pd.read_csv(label_path) if label_path else label
    

    data_preprocessing = Manipulations()
    data_preprocessed = data_preprocessing.manipulate(data = data)
    encoder = MyOneHotEncoder()
    encoder.fit(data = data_preprocessed)
    data_preprocessed_encoded= encoder.transform()
    if label_encoding:
        label_encoded = encoder.label_encoding(label_path = label_path, label = label)
    
    encoder.save()
    engineering = FeatureEngineering()
    engineering = engineering.to_categoric(data = data_preprocessed_encoded)
    data_preprocessed_encoded_engineered = engineering.process()

    return (data_preprocessed_encoded_engineered,label_encoded) if label_encoding else data_preprocessed_encoded_engineered

if __name__ == '__main__':
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data = df.iloc[:, :-1]
    label = df.iloc[:, -1]
    process_data(data_path=None, data=data, label_path = None,label = label,  label_encoding = True)