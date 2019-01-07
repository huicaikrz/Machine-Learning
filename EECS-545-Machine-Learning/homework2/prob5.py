import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

def result_show():
    plt.scatter(data[target > 0][:,0], data[target > 0][:,1],color = 'r',label = '1', s = 10)
    plt.scatter(data[target < 0][:,0], data[target < 0][:,1],color = 'b',label = '-1', s = 10)
    plt.plot(np.linspace(-1,3,100),[-theta[0]/theta[1]*i-theta[2]/theta[1] for i in np.linspace(-1,3,100)])
    plt.legend(loc = 'upper right')
    plt.show()
    
def percepton(n_epochs,X,Y,theta):
    N = len(target)
    for j in range(n_epochs):
        for i in range(N):
            predict = theta.T.dot(data[i])
            if target[i]*predict <= 0:
                theta = theta + target[i]*data[i]           
    return theta
#a
data = np.zeros((100, 3))
val = np.random.uniform(0, 2, 100)
diff = np.random.uniform(-1, 1, 100)
data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
target = np.asarray(val > 1, dtype = int) * 2 - 1    

theta = percepton(10,data,target,np.zeros(3))
result_show()
#b
data = np.ones((100, 3))
data[:50,0], data[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
data[:50,1], data[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
target = np.zeros(100)
target[:50], target[50:] = -1 * np.ones(50), np.ones(50)

theta = percepton(10,data,target,np.zeros(3))
result_show()
#LDA
X1 = data[target > 0][:,0:2]
X2 = data[target < 0][:,0:2]
pi1 = 0.5;pi2 = 0.5
mu1 = X1.mean(axis = 0);mu2 = X2.mean(axis = 0)
sigma = ((X1-mu1).T.dot(X1-mu1)+(X2-mu2).T.dot(X2-mu2))/(len(data)-2)

plt.scatter(X1[:,0], X1[:,1],color = 'r',label = '1', s = 10)
plt.scatter(X2[:,0], X2[:,1],color = 'b',label = '-1', s = 10)

delta = 0.025
x = np.arange(-3, 3, delta);
y = np.arange(-3, 3, delta);
X, Y = np.meshgrid(x, y);
Z1 = mlab.bivariate_normal(X, Y, sigmax=sigma[0][0], sigmay=sigma[1][1], mux=mu1[0], muy=mu1[1], sigmaxy=sigma[0][1]);
Z2 = mlab.bivariate_normal(X, Y, sigmax=sigma[0][0], sigmay=sigma[1][1], mux=mu2[0], muy=mu2[1], sigmaxy=sigma[0][1]);
Z = Z2 - Z1;
plt.contour(X, Y, Z, levels=np.linspace(np.min(Z),np.max(Z),10));
cs = plt.contour(X, Y, Z, levels=[0], c="k", linewidths=3);
plt.clabel(cs, fontsize=10, inline=1, fmt='%1.3f')
plt.title("Countours:  $P(y=1 | x) - P(y=0 | x)$", fontsize=10)
   


