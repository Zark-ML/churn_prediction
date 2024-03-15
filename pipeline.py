from abc import ABC, abstractmethod
from models.decision_tree import DecisionTreeModel
from preprocessing.onehotencoder import OneHotEncoder
from preprocessing.standart_scaler import StandardScaler
from preprocessing.manipulations import Manipulations
from preprocessing.feature_engineering import FeatureEngineering
import pandas as pd
import numpy as np
from typing import Union
from models import select_model


class MlPipeline(ABC):
    """
    Abstract base class for a machine learning pipeline, encapsulating common workflow steps.

    This class provides a framework for creating reusable and modular ML pipelines.
    Child classes must implement abstract methods to customize specific operations.

    Attributes:
        logger (loguru.Logger): A Loguru logger instance for recording messages.
    """

    def __init__(self, model_name: str, preprocessing_steps: dict, feature_selecton_steps: dict, model_params: dict, data: pd.DataFrame, target: pd.Series, data_path: str, target_path,  model_path: str, version: str) -> None:
        """ Initializes the machine learning pipeline with the specified parameters. """
        pass

    @abstractmethod
    def ingest_data(self):
        """
        Ingests data from the specified source.

        This method defines how data is retrieved and loaded into the pipeline.
        Child classes must implement the specific logic for their data source.

        Raises:
            NotImplementedError: This method must be implemented in child classes.
        """

        pass

    @abstractmethod
    def validate_data(self, data):
        """
        Validates the ingested data for quality and consistency.

        This method checks for missing values, outliers, and other data integrity issues.
        Child classes can apply domain-specific checks as needed.

        Args:
            data: The ingested data to be validated.

        Raises:
            ValueError: If data quality issues are detected.
        """

        pass

    @abstractmethod
    def preprocess_data(self, data):
        """
        Preprocesses the data for model training.

        This method typically involves cleaning, transforming, and scaling the data.
        Child classes can implement specific preprocessing techniques as required.

        Args:
            data: The validated data to be preprocessed.

        Returns:
            The preprocessed data.
        """

        pass

    @abstractmethod
    def train_model(self, preprocessed_data):
        """
        Trains a machine learning model using the preprocessed data.

        This method selects, configures, and fits a model to the data.
        Child classes define the specific model and training parameters.

        Args:
            preprocessed_data: The preprocessed data for training.

        Returns:
            The trained model.
        """

        pass

    @abstractmethod
    def analyze_model(self, model, data):
        """
        Analyzes the performance of the trained model.

        This method evaluates the model on a held-out set or evaluates other metrics.
        Child classes should implement evaluation metrics relevant to their task.

        Args:
            model: The trained machine learning model.
            data: The data to be used for evaluation (e.g., test set).

        Returns:
            Evaluation results (e.g., accuracy, F1-score).
        """

        pass

    @abstractmethod
    def version_model(self, model, version):
        """
        Versions the trained model and persists it for later use.

        This method assigns a version identifier and saves the model in a persistent storage.
        Child classes define the versioning scheme and storage mechanism.

        Args:
            model: The trained machine learning model.
            version: The version identifier for the model.
        """

        pass

    @abstractmethod
    def save_model(self, model, version):
        """
        Saves the model to persistent storage for the specified version.

        Args:
            model: The model to be saved.
            version: The version identifier for the model.
        """

        pass

    @abstractmethod
    def load_model(self, version):
        """
        Loads a previously saved model from persistent storage.

        Args:
            version: The version identifier of the model to load.

        Returns:
            The loaded model object.
        """

        pass

    def run(self):
        """
        Executes the complete machine learning pipeline.

        This method orchestrates the different steps of the pipeline:

        1. Data Ingestion
        2. Data Validation
        3. Data Preprocessing
        4. Model Training & Tuning
        5. Model Analysis
        6. Model Versioning

        Logs progress and results using the internal logger.
        """

        pass



# class MainPipeline(MlPipeline):
#     """
#     Concrete implementation of a machine learning pipeline for the Telco Customer Churn dataset.

#     This class extends the abstract MlPipeline class with custom
#     logic for data ingestion, preprocessing, and model training. It uses a DecisionTreeModel
#     as the underlying machine learning model. The pipeline is designed to be reusable and extensible.
#     """
    
#     def __init__(self, 
#                  model_name: Union[str, None] = None, 
#                  preprocessing_steps: Union[dict, None] = None, 
#                  feature_selecton_steps: Union[dict, None] = None, 
#                  model_params: Union[dict, None] = None, 
#                  data: Union[pd.DataFrame, None] = None, 
#                  target: Union[pd.Series, None] = None, 
#                  data_path: Union[str, None] = None, 
#                  target_path: Union[str, None] = None, 
#                  model_path: Union[str, None] = None, 
#                  version: Union[str, None] = None) -> None:

#         """
#         Initializes the machine learning pipeline with the specified parameters.
#         """
#         self.model_name = model_name
#         self.preprocessing_steps = preprocessing_steps
#         self.feature_selecton_steps = feature_selecton_steps
#         self.model_params = model_params
#         self.data = data
#         self.target = target
#         self.data_path = data_path
#         self.target_path = target_path
#         self.model_path = model_path
#         self.version = version

#     def ingest_data(self):
#         if self.data_path & self.target_path:
#             self.data = pd.read_csv(self.data_path)
        

if __name__ == '__main__':
    select_model.SelectModel()()