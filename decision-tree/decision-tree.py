import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def if_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Base cases for stopping tree growth
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select features for splitting
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # Find the best feature and threshold for splitting
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_indices, right_indices = self._split(X[:, best_feat], best_thresh)

        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth+1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth+1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        # Iterate over selected features and find the best split
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                # Update best split if the gain is higher
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Split the data based on the threshold
        left_indices, right_indices = self._split(X_column, threshold)

        # If one of the child nodes is empty, return 0 gain
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
    
        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        e_l, e_r = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        # Split the data based on the threshold
        left_indices = np.argwhere(X_column <= split_thresh).flatten()
        right_indices = np.argwhere(X_column > split_thresh).flatten()
        return left_indices, right_indices

    def _entropy(self, y):
        # Calculate entropy of the labels
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        # Find the most common label in the data
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # Make predictions for the input data
        return np.array([self._transverse_tree(x, self.root) for x in X])

    def _transverse_tree(self, x, node):
        # Traverse the decision tree to make predictions
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._transverse_tree(x, node.left)
        return self._transverse_tree(x, node.right)