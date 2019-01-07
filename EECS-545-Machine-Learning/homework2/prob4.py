import numpy as np
from math import exp
import matplotlib.pyplot as plt

cov_cal = lambda X,Y,sigma: np.array([exp(-(x-y)**2/2/sigma**2) 
                            for x in X for y in Y]).reshape(len(X),len(Y))

def posterior_GP(X,X_train,y_train,sigma):
    cov_given = cov_cal(X_train,X_train,sigma)
    cov_X = cov_cal(X,X,sigma)
    cov_Xgiven = cov_cal(X,X_train,sigma)
    cov_givenX = cov_cal(X_train,X,sigma)
    pos_mu = cov_Xgiven.dot(np.linalg.inv(cov_given)).dot(y_train)
    pos_cov = cov_X  - cov_Xgiven.dot(np.linalg.inv(cov_given)).dot(cov_givenX)
    return pos_mu,pos_cov 

def main():
    X = np.linspace(-5, 5, 100)
    for sigma in [0.3,0.5,1.0]:
        mu = np.zeros(len(X))
        cov = cov_cal(X,X,sigma)
        plt.plot(X,np.random.multivariate_normal(mu,cov,5).T)
        plt.title('sigma = %s' %(sigma))
        plt.show()

    X_train = [-1.3 , 2.4 , -2.5 ,-3.3 , 0.3]
    y_train = [2 , 5.2 , -1.5 , -0.8 , 0.3]
    for sigma in [0.3,0.5,1.0]:
        mu,cov = posterior_GP(X,X_train,y_train,sigma)
        plt.scatter(X_train,y_train,c = 'r')
        plt.plot(X,mu,lw = 3)
        plt.plot(X,np.random.multivariate_normal(mu,cov,5).T,lw = 1)
        plt.title('sigma = %s' %(sigma))
        plt.show()                        

main()