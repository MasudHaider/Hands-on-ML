import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
        errors = (y - output)
        self.w_[1:] += self.eta * X.T.dot(errors)
        self.w_[0] += self.eta * errors.sum()
        cost = (errors ** 2).sum() / 2.0
        self.cost_.append(cost)

        return self

    def net_input(self, X):
        wb = np.dot(X, self.w_[1:])
        withb = wb + self.w_[0]
        return withb

    def predict(self, X):
        return self.net_input(X)


if __name__ == '__main__':
    X = np.array([[2, 16, 8, 5.0],
                  [4, 32, 8, 5.2],
                  [6, 64, 12, 5.5]])
    y = np.array([10000, 12000, 15000])
    from sklearn.preprocessing import StandardScaler

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X) # scale X
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()  # scale y
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)  # train model

    # predict a new phone's price
    mbl_std = sc_x.transform(np.array([[8, 128, 8, 5.1]]))
    price_std = lr.predict(mbl_std)
    print("price: %.3f" % sc_y.inverse_transform(price_std))