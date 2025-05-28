import numpy as np
import cvxpy as cp

''' 
This implements FTRL with centered updates, using the fully adaptive regularizers 
proposed by Orabona (2022), rather than the semi-adaptive ones from McMahan (2017). 

Key differences include:
- The regularizers are not shifted by the Lipschitz constant.
- The fixed \sigma term.
'''

class FTRL:
    def __init__(self, dim, R):
        # problem
        self.dim = dim
        self.x = cp.Variable(self.dim)
        self.x.value = np.zeros(self.dim)
        self.R = R

        # ftrl
        self.constant_sigma = 1 / (4 * R)  # This is my analysis, to use the min trick.
        self.sigmas = []
        self.acc_grads = np.zeros(self.dim)
        self.acc_sigma = 1e-4  # to avoid solving LP for t=1
        self.acc_errors_list = []

        # projection
        self.non_proj_parameter = cp.Parameter(self.dim)
        self.objective = cp.Minimize(cp.sum_squares(self.x - self.non_proj_parameter))
        # constraints = [-1 <= self.x, self.x <= 1, cp.norm(self.x, 2) <= self.R]
        constraints = [cp.norm(self.x, 2) <= self.R]
        # noinspection PyTypeChecker
        self.prob = cp.Problem(self.objective, constraints)

    def get_action(self, grad, pred):
        x_unconst = (- self.acc_grads - pred) / self.acc_sigma

        # Projection
        if np.linalg.norm(x_unconst, 2) + 1e-4 > self.R:
            self.x.value = (x_unconst / (np.linalg.norm(x_unconst, 2)+1e-4)) * self.R
        else:
            self.x.value = x_unconst

        # update params
        error = np.linalg.norm(grad - pred) ** 2
        if not self.acc_errors_list:
            self.acc_errors_list.append(error)
            increment = np.sqrt(self.acc_errors_list[-1])
        else:
            self.acc_errors_list.append(self.acc_errors_list[-1] + error)
            increment = np.sqrt(self.acc_errors_list[-1]) - np.sqrt(self.acc_errors_list[-2])
        self.sigmas.append(self.constant_sigma * increment)
        self.acc_sigma += self.sigmas[-1]
        print("Accumelated sigma: ", self.acc_sigma)

        self.acc_grads += grad
        rounded = np.round(self.x.value, 5)
        return rounded
        # return self.x.value
