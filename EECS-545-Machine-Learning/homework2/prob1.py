import numpy as np
import matplotlib.pyplot as plt
from math import exp

MSE = lambda y,y_fit: ((y-y_fit)**2).mean()

def normalize(X_train,X_test):
    train_mean = X_train[:,1:].mean(axis=0);train_std = X_train[:,1:].std(axis=0)
    X_train[:,1:] = (X_train[:,1:] - train_mean)/train_std
    X_test[:,1:] = (X_test[:,1:] - train_mean)/train_std
    return X_train,X_test

def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)
    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys

def result_show(X_test,y_test,pre_label):
    plt.scatter(X_test[:,1],pre_label,label = 'y_predict',s = 10)
    plt.scatter(X_test[:,1],y_test,label = 'y_test',s = 10)
    plt.legend(loc = 'upper left')
    plt.xlabel('x');plt.ylabel('label')
    plt.show()
    print('The test error: ', MSE(y_test,pre_label))
    
def main():
    noise_scale = 0.05

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)
    X_train = np.column_stack((np.array([1]*len(X_train)),X_train))
    X_test = np.column_stack((np.array([1]*len(X_test)),X_test))
    X_train,X_test = normalize(X_train,X_test)
    
    theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    pre_label = X_test.dot(theta)
    result_show(X_test,y_test,pre_label)
        
    # bandwidth parameters
    sigma_paras = [0.2,2.0]
    for tau in sigma_paras:
        pre_label = []
        for i in range(len(y_test)):
            R = np.diag([exp(-(X_test[i,1]-xi)**2/2/tau**2) for xi in X_train[:,1]])
            theta = np.linalg.inv(X_train.T.dot(R).dot(X_train)).\
                                  dot(X_train.T).dot(R).dot(y_train)
            pre_label.append(X_test[i,:].T.dot(theta))
        result_show(X_test,y_test,pre_label)
        
main()