from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

class Metrics:
    def __init__(self, y_true, y_pred):
        """
        Constructor for the Metrics class.

        Parameters:
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def roc_auc(self):
        """
        Calculate the ROC AUC (Receiver Operating Characteristic - Area Under the Curve) score.

        Returns:
        - float: ROC AUC score
        """
        return roc_auc_score(self.y_true, self.y_pred)

    def f1_score(self, average='macro'):
        """
        Calculate the F1 Score.

        Parameters:
        - average: str, optional, averaging strategy for multi-class classification
                   (default is 'macro')

        Returns:
        - float: F1 Score
        """
        return f1_score(self.y_true, self.y_pred, average=average)

    def p4_score(self):
        """
        Calculate the P4 Score.

        Returns:
        - float: P4 Score
        """
        tn, fp, fn, tp = np.ravel(confusion_matrix(self.y_true, self.y_pred))
        try:
            return 4 * tp * tn / (4 * tp * tn + (tp + tn) * (fp + fn))
        except ZeroDivisionError:
            return 0

    def confusion_matrix(self):
        """
        Generate the confusion matrix.

        Returns:
        - array: Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_matrix(self):
        """
        Plot and save the confusion matrix as an image.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")

    def save_metrics(self, filename):
        """
        Save the calculated metrics to a JSON file.

        Parameters:
        - filename: str, name of the file to save metrics
        """
        roc_auc = self.roc_auc()
        f1 = self.f1_score()
        p4 = self.p4_score()
        results_dict = {
            "ROC AUC": roc_auc,
            "F1 Score": f1,
            "P4 Score": p4,
            "Confusion Matrix": np.ravel(confusion_matrix(self.y_true, self.y_pred)).tolist()
        }
        with open(filename, 'w') as file:
            json.dump(results_dict, file)


# if __name__=="__main__":
#     y_true = np.array([0, 1, 0, 1, 1])
#     y_pred = np.array([1, 1, 1, 0, 0])
#     metrics = Metrics(y_true, y_pred)
#     roc_auc = metrics.roc_auc()
#     f1 = metrics.f1_score()
#     p4 = metrics.p4_score()
#     cm = metrics.confusion_matrix()
#     print("ROC AUC:", roc_auc)
#     print("F1 Score:", f1)
#     print("P4 Score:", p4)
#     print("Confusion Matrix:")
#     print(cm)
#     metrics.save_metrics("metrics.json")