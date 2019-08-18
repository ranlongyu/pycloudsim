import numpy as np
import numpy.random as nr

a = [1,2,3,10]
print(np.clip(np.random.normal(a, 0.5), -2, 2))

exit()

class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        #  2.5 * np.random.randn(2, 4) + 3 #生成2行4列的符合（3,2.5^2）正态分布矩阵 其中u=3 σ=2.5
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(10):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
