import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mtn

def plot_contour():
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
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    plt.show()

sigma_extract = lambda sigma,indice1,indice2: np.array([sigma[i][j] 
                for i in indice1 for j in indice2]).reshape(len(indice1),len(indice2))

def marginal_for_guassian(sigma,mu,given_indices):
    mu_vec = np.array([mu[i] for i in given_indices])
    sigma_mat = sigma_extract(sigma,given_indices,given_indices)
    return mu_vec,sigma_mat
   
def conditional_for_gaussian(sigma,mu,given_indices,given_values):
    remain_indices = [i for i in range(0,len(mu)) if i not in given_indices]    
    mu_remain = np.array([mu[i] for i in remain_indices])
    mu_given = np.array([mu[i] for i in given_indices])    
    sigma_given = sigma_extract(sigma,given_indices,given_indices)  
    sigma_remain = sigma_extract(sigma,remain_indices, remain_indices)
    sigma_gire = sigma_extract(sigma,given_indices,remain_indices)
    sigma_regi = sigma_extract(sigma,remain_indices,given_indices)
    sigma_cond = sigma_remain - sigma_regi.dot(np.linalg.inv(sigma_given)).dot(sigma_gire)    
    mu_cond = mu_remain + sigma_regi.dot(np.linalg.inv(sigma_given)).dot(given_values-mu_given)
    return mu_cond,sigma_cond
#initialize raw data for b,d
test_sigma_1 = np.array(
    [[1.0, 0.5],
     [0.5, 1.0]]
)

test_mu_1 = np.array(
    [0.0, 0.0]
)

test_sigma_2 = np.array(
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 1.5],
     [0.0, 0.0, 2.0, 0.0],
     [0.0, 1.5, 0.0, 4.0]]
)

test_mu_2 = np.array(
    [0.5, 0.0, -0.5, 0.0]
)

indices_1 = np.array([0])

indices_2 = np.array([1,2])
values_2 = np.array([0.1,-0.2])
#b
mu,sigma = marginal_for_guassian(test_sigma_1, test_mu_1, indices_1)
print('mean is',mu);print('covariance is ', sigma)
plt.plot(np.linspace(-5,5,200),[mtn.pdf(x, mean=mu, cov=sigma) for x in np.linspace(-5,5,200)])
plt.xlabel('x');plt.ylabel('probability density function')
plt.show()
#d
conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)
mu,sigma = conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)
print('mean is',mu);print('covariance is ', sigma)
plot_contour()
