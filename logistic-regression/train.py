import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

reg = LogisticRegression(lr=0.0001, n_iters=1000)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)


def accuracy(y_test, predictions):
    """
    The accuracy_score function computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions.
    In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    """
    accuracy = np.sum(y_test == predictions) / len(y_test)
    return accuracy 

acc = accuracy(y_test, predictions)
print(acc)