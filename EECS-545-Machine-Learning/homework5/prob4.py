import numpy as np
np.random.seed(1234)

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mtn

def E_step(X_train,means,covs,pis):
    N = len(X_train)
    r = np.zeros(K*N).reshape(N,K)
    for m in range(K):
        mu = means[m]
        sigma = covs[m]
        pi = pis[m]
        #e-step    
        for i in range(N):    
            r[i,m] = pi*mtn.pdf(X_train[i], mean=mu, cov=sigma)
    total = np.cumsum(r,axis = 1)[:,-1]
    for m in range(K): 
        r[:,m] = r[:,m]/total
    return r
    
def M_step(r,X_train):
    N = len(X_train)
    for m in range(K):
        means[m] = list(np.average(X_train,weights = r[:,m],axis =0))
        sdX_train = X_train-means[m]
        covs[m] = (np.array(list(r[:,m])*2).reshape(2,N) * sdX_train.T).dot(sdX_train)/r[:,m].sum()
        #covs[m] = np.average(np.array([nX_train[i].reshape(2,1).dot(nX_train[i].reshape(1,2)) for i in range(N)]),axis = 0,weights = r[:,m])
        pis[m] = r[:,m].sum()/N
    return means,covs,pis
    
def random_posdef(n):
  A = np.random.rand(n, n)
  return np.dot(A, A.transpose())

# Parameter initialization ###
K = 10
pis = [1.0/K for i in range(K)]
means = [[0,0] for i in range(K)]
covs = [random_posdef(2) for i in range(K)]
##############################
import os
os.chdir("C:/Users/lenovo/Desktop/machine learning/EECS 545/homework5") 
X_train = np.load("gmm_data.npy")

Ks = [2,3,5,10]

n_epochs = 50 
for i in range(n_epochs):
    r = E_step(X_train,means,covs,pis)
    means,covs,pis = M_step(r,X_train)
    #if i in [0,4,9,19,49]:
     #   GMM_contour()
#b
def GMM_contour():
    x = np.linspace(-1., 6.)
    y = np.linspace(0., 8.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T   
    Z = map(lambda x: sum([pis[m]*mtn.pdf(x, mean=means[m], cov=covs[m]) for m in range(K)]),XX) 
    Z = np.array(list(Z)).reshape(X.shape)
    plt.contour(X, Y, Z,levels = np.linspace(Z.min(),Z.max(),10))
    plt.scatter(X_train[:, 0], X_train[:, 1],0.8)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()

GMM_contour()
  

    