#This code realize the perceptron kernel method. I strictly follow the outline
#in homwwork. However, the existence of yi in w doest not make sense to me.
#At the very beginning, I realize my own version of kernel method, which gives
#the same output. As for this code, for the purpose of understanding and stick 
#the algorithm in writing part, it's actually computational burden. (calculate 
#kernel again and again). I will further research if there's difference in my 
#understanding and the given hint.
import numpy as np
import matplotlib.pyplot as plt

kernel_cal = lambda X,y,sigma: np.array([np.exp(-np.linalg.norm(x-y)**2/2/sigma**2) 
                             for x in X]).reshape(len(X),1)

predict_label = lambda alpha,Y,X,xnew: alpha.T.dot(Y*kernel_cal(X,xnew,sigma))
gradient_cal = lambda alpha,Y,X,xnew: Y*kernel_cal(X,xnew,sigma)

def perKernel_SGD(n_epochs,alpha,X,Y):
    #X is raw featyre,Y is label
    order = list(range(len(Y)))
    np.random.shuffle(order)
    for k in range(n_epochs):
        for j in range(len(Y)):
            #train data j
            if Y[j] *predict_label(alpha,Y,X,X[j]) <= 0:
                alpha += Y[j]*gradient_cal(alpha,Y,X,X[j])
    print(sum((np.array([predict_label(alpha,Y,X,X[j]) for j in range(len(Y))]).reshape(len(Y),1)*Y)>0)/len(Y))
    return alpha

def result_show(X,Y,sigma):
    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    Z = np.array([predict_label(alpha,Y,X,np.array([x[j][k],y[j][k]]).reshape(1,2)) 
            for j in range(len(x)) for k in range(len(x))]).reshape(len(x),len(x))
    cs = plt.contour(x, y, Z,levels = [0], c="k", linewidths=1);
    plt.clabel(cs, fontsize=10, inline=1, fmt='%1.3f')
    plt.title("decision boundary for sigma = %s" %(sigma),fontsize=10)
    plt.legend(loc = 'upper right')
    plt.show()

np.random.seed(17)
data = np.ones((100, 2))
data[:,0] = np.random.uniform(-1.5, 1.5, 100)
data[:,1] = np.random.uniform(-2, 2, 100)
z = data[:,0] ** 2 + ( data[:,1] - (data[:,0] ** 2) ** 0.333 ) ** 2  
target = np.asarray( z > 1.5, dtype = int) * 2 - 1

for sigma in [0.1,1.0]:
    N = len(data)
    alpha = np.zeros(N).reshape(N,1)
    alpha = perKernel_SGD(30,alpha,data,target.reshape(len(target),1))
    plt.scatter(data[target > 0][:,0], data[target > 0][:,1],color = 'r',label = '1', s = 10)
    plt.scatter(data[target < 0][:,0], data[target < 0][:,1],color = 'b',label = '-1', s = 10)
    result_show(data,target.reshape(len(target),1),sigma)
    

