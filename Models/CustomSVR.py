from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel
from scipy import sparse
import osqp
import numpy as np

def weight_data_mmd(source_data, target_data, gamma=100):
    n_source = source_data.shape[0]
    n_target = target_data.shape[0]

    ss_pairwise = rbf_kernel(X=source_data, gamma=gamma)
    tt_pairwise = rbf_kernel(X=target_data, gamma=gamma)
    st_pairwise = rbf_kernel(X=source_data, Y=target_data, gamma=gamma)


    # Define problem data
    P = sparse.csc_matrix(ss_pairwise / (n_source * n_source))
    q = np.sum(st_pairwise, axis=1) / (n_source * n_target) * (-1)
    A = sparse.csc_matrix(np.identity(n_source))
    l = np.zeros((n_source, ))
    u = np.ones((n_source, )) * np.inf

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.0)

    # Solve problem
    res = prob.solve()
    v = res.x
    mmd = np.matmul(np.matmul(v.T, ss_pairwise), v) - 2 * np.sum(np.matmul(st_pairwise.T, v)) + np.sum(tt_pairwise)
    mmd = float(np.sqrt(mmd))
    return res, mmd

def SVR_decision_boundary(model):
    support_vectors = model.support_vectors_
    alpha = model.dual_coef_
    predict = model.predict(support_vectors)
    tmp = predict.T * alpha
    boundary = support_vectors * (tmp.T)
    boundary = np.sum(boundary, axis=0)
    omega = np.linalg.norm(boundary)
    return omega

class CustomSVR():
    def __init__(self, first_model=SVR):
        self.svr = first_model(gamma=0.01)
        # self.linear = second_model()

    def fit(self, X, y, sample_weight=None):
        self.svr.fit(X, y.ravel(), sample_weight=sample_weight)
        self.omega = SVR_decision_boundary(self.svr)
        # pred = self.svr.predict(X).ravel()
        # true = y
        # self.linear.fit(pred.reshape(-1, 1), true)

    def predict(self, X):
        pred = self.svr.predict(X).flatten()
        # y = self.linear.predict(pred.ravel().reshape(-1, 1))
        return pred

    def distance_boundary(self, predict_value):
        return abs(predict_value) / self.omega