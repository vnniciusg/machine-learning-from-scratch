import numpy as np

def sigmoid(x):
    """
    The sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
    It is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
    A common choice for the sigmoid function is the logistic function:
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0   

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias # linear_pred = X * weights + bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) # dw = (1 / n_samples) * X.T * (predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y) # db = (1 / n_samples) * Σ(predictions - y)

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        predictions_cls =  [0 if y<=0.5 else 1 for y in y_pred]
        return predictions_cls