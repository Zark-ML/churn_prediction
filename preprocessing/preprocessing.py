from abc import ABC, abstractmethod
from loguru import logger

class AbstractPreprocessor(ABC):
    """
    Abstract base class for data preprocessing, encapsulating common patterns and enforcing structure.

    This class provides a framework for creating reusable and modular data preprocessing steps.
    Child classes must implement abstract methods to customize specific preprocessing tasks.

    Attributes:
        logger (loguru.Logger): A Loguru logger instance for recording messages.
    """

    def __init__(self):
        """
        Initializes the preprocessor, creating a default Loguru logger with the class name.
        """
        self.logger = logger(__name__)

    @abstractmethod
    def fit(self, data):
        """
        Fits the preprocessor to the data, performing any necessary calculations or adjustments.

        This method may be used to learn parameters, compute statistics, or prepare internal data
        structures based on the provided data.

        Args:
            data: The data to be fit to the preprocessor. The specific data type may vary
                depending on the child class implementation.

        Raises:
            NotImplementedError: This method must be implemented in child classes.
        """

        raise NotImplementedError("This method must be implemented in child classes")

    @abstractmethod
    def transform(self, data):
        """
        Transforms the data using the fitted preprocessor.

        This method applies the learned parameters or internal data structures to transform the input
        data according to the child class's specific preprocessing tasks.

        Args:
            data: The data to be transformed. The specific data type may vary depending on the
                child class implementation.

        Returns:
            The transformed data. The return type may vary depending on the child class
                implementation.

        Raises:
            NotImplementedError: This method must be implemented in child classes.
        """

        raise NotImplementedError("This method must be implemented in child classes")

    def fit_transform(self, data):
        """
        Fits the preprocessor to the data and then transforms it.

        This is a convenience method that combines the `fit` and `transform` steps into one.

        Args:
            data: The data to be fit and transformed. The specific data type may vary depending
                on the child class implementation.

        Returns:
            The transformed data. The return type may vary depending on the child class
                implementation.
        """

        self.fit(data)
        return self.transform(data)

    def _log_info(self, message):
        """
        Logs an informational message using the logger.
        """
        self.logger.info(message)

    def _log_debug(self, message):
        """
        Logs a debug message using the logger.
        """
        self.logger.debug(message)

    def _log_warning(self, message):
        """
        Logs a warning message using the logger.
        """
        self.logger.warning(message)

    def _log_error(self, message, exception=None):
        """
        Logs an error message using the logger.

        Args:
            message (str): The message to be logged.
            exception (Exception, optional): An optional exception object to log.
        """
        if exception:
            self.logger.error(message, exc_info=exception)
        else:
            self.logger.error(message)