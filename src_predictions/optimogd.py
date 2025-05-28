import numpy as np
import cvxpy as cp


class OOGD:
    def __init__(self, dim, R):
        # problem
        self.dim = dim
        self.x = cp.Variable(self.dim)
        self.x.value = np.zeros(self.dim)

        self.x_inter = cp.Variable(self.dim)
        self.x_inter.value = np.zeros(self.dim)

        self.R = R

        # omd
        self.actions = []

        self.inter_actions = []

        self.grads = []
        self.acc_grad_norms = []
        self.eta = 0

        # projection
        self.non_proj_parameter = cp.Parameter(self.dim)
        self.objective = cp.Minimize(cp.sum_squares(self.x - self.non_proj_parameter))
        # constraints = [-1 <= self.x, self.x <= 1, cp.norm(self.x,2) <= self.R]
        constraints = [cp.norm(self.x, 2) <= self.R]
        # noinspection PyTypeChecker
        self.prob = cp.Problem(self.objective, constraints)

    def get_action(self, grad, pred):
        if not self.actions:
            first_action = -pred / np.linalg.norm(pred, 2) * self.R
            first_inter_action = -grad / np.linalg.norm(grad, 2) * self.R
            error = np.linalg.norm(grad - pred) ** 2

            self.actions.append(first_action)
            self.inter_actions.append(first_inter_action)

            self.grads.append(grad)
            self.acc_grad_norms.append(error)
            return np.zeros(self.dim)

        x_unconst_optimistic = self.inter_actions[-1] - self.eta * pred
        # closed form projection:
        if np.linalg.norm(x_unconst_optimistic, 2)+1e-4 > self.R:
            self.x.value = (x_unconst_optimistic / (np.linalg.norm(x_unconst_optimistic, 2)+1e-4)) * self.R
        else:
            self.x.value = x_unconst_optimistic

        # update params
        # updating intermediate action
        x_unconst_inter_optimistic = self.inter_actions[-1] - self.eta * self.grads[-1]
        # Projection
        # closed form projection:
        if np.linalg.norm(x_unconst_inter_optimistic, 2)+1e-4 > self.R:
            self.x_inter.value = (x_unconst_inter_optimistic / (np.linalg.norm(x_unconst_inter_optimistic, 2)+1e-4)) * self.R
        else:
            self.x_inter.value = x_unconst_inter_optimistic

        self.inter_actions.append(self.x_inter.value)
        ##############################

        self.actions.append(self.x.value)
        self.grads.append(grad)
        error = np.linalg.norm(grad - pred) ** 2
        # self.acc_grad_norms.append(self.acc_grad_norms[-1] + np.linalg.norm(grad, 2)**2)
        self.acc_grad_norms.append(self.acc_grad_norms[-1] + error)
        self.eta = (2 * self.R) / (np.sqrt(2) * np.sqrt(self.acc_grad_norms[-1])+2)  # Orabona 4.14, 2R is D
        rounded = np.round(self.x.value, 4)

        return rounded
