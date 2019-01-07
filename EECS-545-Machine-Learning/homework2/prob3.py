import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mtn

class bayesian_linear_regression(object):
    def __init__(self,sigma,mu,beta):
        self.sigma = sigma
        self.mu = mu
        self.beta = beta

    def update(self,phi,t):
        sigma_inv = np.linalg.inv(self.sigma) 
        self.sigma = np.linalg.inv((sigma_inv+ self.beta*phi.T.dot(phi)))
        self.mu = self.sigma.dot(self.beta*phi.T.dot(t)+sigma_inv.dot(self.mu))
        
def data_generator(size,scale):
    x = np.random.uniform(low=-3, high=3, size=size)
    rand = np.random.normal(0, scale=scale, size=size)
    y = 0.5 * x - 0.3 + rand
    phi = np.array([[x[i], 1] for i in range(x.shape[0])])
    t = y
    return phi, t

def plot_heat(mu,sigma):
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = mtn.pdf((X[j][i],Y[j][i]), mean=mu, cov=sigma)
    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none', extent=[-3.0, 3.0, -3.0, 3.0],cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()

def main():
    alpha = 2
    sigma_0 = np.diag(1.0/alpha*np.ones([2]))
    mu_0 = np.zeros([2])
    beta = 1.0
    blr_learner = bayesian_linear_regression(sigma_0, mu_0, beta=beta)
    
    show = [0,1,10,20]
    num_episodes = 21
    for epi in range(num_episodes):
        if epi in show:
            print('The mean of w: ',blr_learner.mu)
            print('The covariance of w: ',blr_learner.sigma)
            plot_heat(blr_learner.mu,blr_learner.sigma)    
        phi, t = data_generator(1,1.0/beta)
        blr_learner.update(phi,t)

main()