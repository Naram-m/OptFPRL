import numpy as np
# The different scenarios

class Env:
    def __init__(self, T, dim):
        self.T = T
        self.dim = dim
        self.t = -1
        self.coefficients = np.ones((self.T, self.dim))
        # customizing costs

    def set_1(self):
        self.coefficients[0:1000] *= -1

    def set_2(self):
        self.coefficients[0:1000] *= -1
        self.coefficients[2000:2500] *= -1
        self.coefficients[3500:3750] *= -1

    def set_3(self):
        self.coefficients[0:1000] *= -1
        self.coefficients[2000:2500] *= -5
        self.coefficients[3500:3750] *= -10

    def set_4(self):
        for i in range(0, self.T, 100):
            self.coefficients[i:i + 50] *= -1

    def set_5(self):
        for i in range(0, self.T, 100):
            self.coefficients[i:i + 50] *= -0.1

