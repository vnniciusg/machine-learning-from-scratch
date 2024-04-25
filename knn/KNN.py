import numpy as np
from typing import List
from collections import Counter


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
        Calculates the Euclidean distance between two points.

        Parameters:
            x1 (np.ndarray): First point.
            x2 (np.ndarray): Second point.

        Returns:
            float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k: int = 3) -> None:
        """
        Initializes the KNN classifeier.
        Parameters:
            k (int) : Number of neighbors to consider. Default is 3.
        """
        self.k = k
        self.X_train = None  # Training input data.
        self.y_train = None  # Training outpu labels.

    def fit(self, X: np.array, y: List[int]) -> None:
        """
        Trains the KNN classifier with the given training data.
        
        Parameters:
                X (np.array): Training input data
                y (List[int] : Training output lables.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.array) -> List[int]:
        """
         Method to predict the class of the given input samples
         
         Parameters:
                    X  (np.ndarray): Augmented coefficient matrix.
        Returns:
                   List[int]: Predicted class labels for each input sample.
        """
        return [self._predict(x) for x in X]

    def _predict(self, x: np.ndarray) -> int:
        """
        Private method to predict class from a single input sample
        Returns:
             result ( int ) : predict class
        """
        # Calculate distances between the input sample and all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Select the indices of the k closest samples
        k_indices = np.argsort(distances)[:self.k]

        # Get the classes corresponding to the k closest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Counts the occurrence of each class and returns the most common class
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
