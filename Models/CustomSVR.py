from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel
from scipy import sparse
import osqp
import numpy as np

class CustomSVR():
    def __init__(self, first_model=SVR):
        self.svr = first_model(gamma=0.01)
        # self.linear = second_model()

    def fit(self, X, y, sample_weight=None):
        self.svr.fit(X, y.ravel(), sample_weight=sample_weight)
        # pred = self.svr.predict(X).ravel()
        # true = y
        # self.linear.fit(pred.reshape(-1, 1), true)

    def predict(self, X):
        pred = self.svr.predict(X).flatten()
        # y = self.linear.predict(pred.ravel().reshape(-1, 1))
        return pred