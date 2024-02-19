from abc import ABC, abstractmethod

class AbstractPreprocessor(ABC):
    """
    Abstract base class for data preprocessing, encapsulating common patterns and enforcing structure.

    This class provides a framework for creating reusable and modular data preprocessing steps.
    Child classes must implement abstract methods to customize specific preprocessing tasks.
    """

    def __init__(self):
        """
        Initializes the preprocessor
        """
        pass
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
        pass
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
        pass

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
