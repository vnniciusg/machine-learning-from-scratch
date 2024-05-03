import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import LinearRegression


# n_samples: The number of samples.
# n_features: The number of features.
# noise: The standard deviation of the gaussian noise applied to the output.
# random_state: The seed used by the random number generator.
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)


def mse(y_test, predictions):
    """ 
    The mean_squared_error function computes mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss.
    It is a risk function, corresponding to the expected value of the squared error loss.
    If y_test is the true target value and y_pred is the corresponding prediction, then the mean squared error (MSE) estimated over n_samples is defined as
    MSE = (1/n_samples) * Σ(y_test - y_pred)^2
    """
    return np.mean((y_test - predictions) ** 2)


mse = mse(y_test, predictions)
print(mse)
