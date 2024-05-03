import numpy as np

class LinearRegression:
    def __init__(self, lr= 0.001, n_iters=1000) -> None:
        """
        Linear Regression model
        lr: learning rate
        n_iters: number of iterations
        """
        self.lr = lr 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
    
    def fit(self, X, y):
        """
        Fit the training data
        X: training data
        y: target variable
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_prediction = np.dot(X, self.weights) + self.bias # y = w*x + b

            dw = (1/n_samples) * np.dot(X.T, (y_prediction - y)) # derivative of weights
            db = (1/n_samples) * np.sum(y_prediction - y) # derivative of bias

            self.weights = self.weights - self.lr * dw # update weights
            self.bias = self.bias - self.lr * db # update bias

    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction

        