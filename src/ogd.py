import numpy as np
import cvxpy as cp


class OGD:
    def __init__(self, dim, R):
        # problem
        self.dim = dim
        self.x = cp.Variable(self.dim)
        self.x.value = np.zeros(self.dim)
        self.R = R

        # omd
        self.actions = []
        self.grads = []
        self.acc_grad_norms = []
        self.eta = 0

        # projection
        self.non_proj_parameter = cp.Parameter(self.dim)
        self.objective = cp.Minimize(cp.sum_squares(self.x - self.non_proj_parameter))
        constraints = [cp.norm(self.x, 2) <= self.R]
        # noinspection PyTypeChecker
        self.prob = cp.Problem(self.objective, constraints)

    def get_action(self, grad):
        if not self.actions:
            self.actions.append(self.x.value)
            self.grads.append(grad)
            self.acc_grad_norms.append(np.linalg.norm(grad,2))
            return np.zeros(self.dim)

        x_unconst = self.actions[-1] - self.eta * self.grads[-1]
        # closed form projection:
        if np.linalg.norm(x_unconst, 2)+1e-4 > self.R:
            self.x.value = (x_unconst / (np.linalg.norm(x_unconst, 2)+1e-4)) * self.R
        else:
            self.x.value = x_unconst

        # update params
        self.actions.append(self.x.value)
        self.grads.append(grad)
        self.acc_grad_norms.append(self.acc_grad_norms[-1] + np.linalg.norm(grad, 2)**2)
        self.eta = (2 * self.R) / (np.sqrt(2) * np.sqrt(self.acc_grad_norms[-1]))  # Orabona 4.14, 2R is D
        rounded = np.round(self.x.value, 4)
        print("OGD:", rounded)

        return rounded