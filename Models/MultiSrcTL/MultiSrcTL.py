import osqp
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import os
import pickle

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

class CustomMultiSrcTL():
    def __init__(self, src_path, src_data, src_mmd, BETA=0.1, SIGMA=0.001, p=1, conf_base=0.06, confidence=0.5):
        self.src_path = src_path
        self.src_data = src_data
        self.src_mmd = src_mmd
        self.n_domain = len(src_data)

        self.BETA = BETA
        self.SIGMA = SIGMA
        self.p = p
        self.conf_base = conf_base
        self.confidence = confidence


    def preProcess(self):
        self.src_model = {}
        self.src_name = {}
        i = 0
        for file in os.listdir(self.src_path):
            net = pickle.load(open(self.src_path + '/' + file, 'rb'))
            self.src_model[file] = net
            self.src_name[i] = file
            i += 1

    def empirical_error(self, src_i, src_j):
        net_i = self.src_model[src_i]
        net_j = self.src_model[src_j]

        # 
        y_pred = net_j.predict(self.src_data[src_i]['input-data'])
        y_true = self.src_data[src_i]['output-data']
        em_error = r2_score(y_true=y_true, y_pred=y_pred)
        return em_error * self.BETA

    def compute_inter_src_relation_matrix(self):
        error_matrix = np.zeros((self.n_domain, self.n_domain))
        for i in range(self.n_domain):
            for j in range(self.n_domain):
                # all empirical_error will be exp
                if i == j:
                    error_matrix[i][j] = 1 # exp(0)
                else:
                    src_i = self.src_name[i]
                    src_j = self.src_name[j]
                    error_matrix[i][j] = np.exp(self.empirical_error(src_i, src_j))

        relation_matrix = np.zeros((self.n_domain, self.n_domain))
        for i in range(self.n_domain):
            for j in range(self.n_domain):
                if i == j:
                    continue
                else:
                    relation_matrix[i, j] = error_matrix[i, j] / (np.sum(error_matrix[i, :]) - 1)
        self.source_matrix = relation_matrix

    def compute_source_target_relation(self, SIGMA=0.001, p=1):
        st_sim = np.zeros(shape=(self.n_domain, ))
        for i in range(self.n_domain):
            name = self.src_name[i]
            st_sim[i, ] = self.src_mmd[name]

        st_sim = np.exp(np.power(st_sim, p) * SIGMA * -1)
        st_sum = np.sum(st_sim)
        st_sim = st_sim / (st_sum + 1e-6)
        self.st_sim = st_sim

    def compute_source_weight(self):
        # self.SIGMA = 1 / self.n_domain
        result = self.SIGMA * np.identity(self.n_domain) + (1 - self.SIGMA) * self.source_matrix
        result = np.matmul(self.st_sim.T, result)
        self.source_weight = result

    def source_domain_predict(self, data):
        predict_list = []
        for i in range(self.n_domain):
            name = self.src_name[i]
            predict_source_i = self.src_model[name].predict(data)
            predict_list.append(predict_source_i)
        return predict_list

    def fit(self, X, y):
        self.preProcess()
        self.compute_inter_src_relation_matrix()
        self.compute_source_target_relation()
        self.compute_source_weight()
        train_pred = self.first_predict(X).ravel().reshape(-1, 1)
        self.final_scaler = LinearRegression()
        self.final_scaler.fit(train_pred, y.ravel())


    def first_predict(self, test_data):
        predict_test_data = []
        n_test = test_data.shape[0]
        predict_values = self.source_domain_predict(data=test_data)
        for i in range(n_test):
            domain_learn = np.zeros((self.n_domain, ))
            predict_value = np.zeros((self.n_domain, ))
            for t in range(self.n_domain):
                predict_value[t] = predict_values[t][i]
            for k in range(self.n_domain):
                # use self.confidence interval instead
                name = self.src_name[k]
                model = self.src_model[name]
                conf = model.distance_boundary(predict_value[k])
                conf = self.conf_base / (self.conf_base + conf)
                # print(conf)
                if conf < self.confidence:
                    domain_learn[k] = (np.matmul(self.source_matrix[k, :].T, predict_value)
                                      - self.source_matrix[k, k] * predict_value[k])
                else:
                    domain_learn[k] = predict_value[k]
            
            result = np.sum(self.source_weight * domain_learn)
            predict_test_data.append(result)
        
        first_pred = np.array(predict_test_data).ravel()
        return first_pred

    def predict(self, test_data):
        pred = self.final_scaler.predict(self.first_predict(test_data).reshape(-1, 1)).reshape(-1, 1)
        return pred