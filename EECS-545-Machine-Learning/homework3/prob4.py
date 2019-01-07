from sklearn.datasets import load_iris
iris=load_iris()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

pi1 = 0.5;pi2 = 0.5
def plot(sigma0,sigma1):
    plt.scatter(data_0[:,0], data_0[:,1],color = 'r',label = '0', s = 10)
    plt.scatter(data_1[:,0], data_1[:,1],color = 'b',label = '1', s = 10)
    x, y = np.mgrid[1.5:5:.01, 0:5.5:.01]
    pos = np.empty(x.shape + (2,));pos[:, :, 0] = x; pos[:, :, 1] = y
    Z = multivariate_normal(mu0, sigma0).pdf(pos) - multivariate_normal(mu1, sigma1).pdf(pos)
    plt.contour(x, y,Z, levels=np.linspace(np.min(Z),np.max(Z),10))
    cs = plt.contour(x, y, Z, levels=[0], c="k", linewidths=1);
    plt.clabel(cs, fontsize=10, inline=1, fmt='%1.3f')
    plt.title("Countours:  $P(y=0 | x) - P(y=1 | x)$", fontsize=10)
    plt.legend(loc = 'upper right')
    plt.show()

#LDA
mu0 = data_0.mean(axis = 0);mu1 = data_1.mean(axis = 0)
sigma = ((data_0-mu0).T.dot(data_0-mu0)+(data_1-mu1).T.dot(data_1-mu1))/(len(data_0)+len(data_1)-2)
plot(sigma,sigma)
#QDA
mu0 = data_0.mean(axis = 0);mu1 = data_1.mean(axis = 0)
sigma0 = (data_0-mu0).T.dot(data_0-mu0)/(len(data_0)-1)
sigma1 = ((data_1-mu1).T.dot(data_1-mu1))/(len(data_1)-1)
plot(sigma0,sigma1)



