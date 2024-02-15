from abc import ABC, abstractmethod
from loguru import logger

class MlPipeline(ABC):
    """
    Abstract base class for a machine learning pipeline, encapsulating common workflow steps.

    This class provides a framework for creating reusable and modular ML pipelines.
    Child classes must implement abstract methods to customize specific operations.

    Attributes:
        logger (loguru.Logger): A Loguru logger instance for recording messages.
    """

    def __init__(self):
        """
        Initializes the pipeline, creating a default Loguru logger with the class name.
        """
        self.logger = logger(__name__)

    @abstractmethod
    def ingest_data(self):
        """
        Ingests data from the specified source.

        This method defines how data is retrieved and loaded into the pipeline.
        Child classes must implement the specific logic for their data source.

        Raises:
            NotImplementedError: This method must be implemented in child classes.
        """

        raise NotImplementedError("This method must be implemented in child classes")

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

        raise NotImplementedError("This method must be implemented in child classes")

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

        raise NotImplementedError("This method must be implemented in child classes")

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

        raise NotImplementedError("This method must be implemented in child classes")

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

        raise NotImplementedError("This method must be implemented in child classes")

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

        raise NotImplementedError("This method must be implemented in child classes")

    @abstractmethod
    def save_model(self, model, version):
        """
        Saves the model to persistent storage for the specified version.

        Args:
            model: The model to be saved.
            version: The version identifier for the model.
        """

        raise NotImplementedError("This method must be implemented in child classes")

    @abstractmethod
    def load_model(self, version):
        """
        Loads a previously saved model from persistent storage.

        Args:
            version: The version identifier of the model to load.

        Returns:
            The loaded model object.
        """

        raise NotImplementedError("This method must be implemented in child classes")
    
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

        try:
            self.logger.info("Starting ML Pipeline...")

            data = self.ingest_data()
            self.logger.info("Data ingested successfully.")

            self.validate_data(data)
            self.logger.info("Data validated successfully.")

            preprocessed_data = self.preprocess_data(data)
            self.logger.info("Data preprocessed successfully.")

            model = self.train_model(preprocessed_data) or self.load_model('version')
            self.logger.info("Model trained successfully.")

            analysis_results = self.analyze_model(model, data)
            self.logger.info("Model analyzed successfully:")
            for metric, value in analysis_results.items():
                self.logger.info(f"- {metric}: {value}")

            self.version_model(model, version=input('Enter Version: '))
            self.logger.info("Model versioned successfully.")

            self.logger.info("ML Pipeline completed successfully!")
        except Exception as e:
            self.logger.error("An error occurred during the pipeline:", exc_info=e)
        finally:
            pass