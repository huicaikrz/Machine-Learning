import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from math import exp,log
from sklearn.model_selection import train_test_split

def normalize(X_train,X_test):
    train_mean = X_train[:,1:].mean(axis=0);train_std = X_train[:,1:].std(axis=0)
    X_train[:,1:] = (X_train[:,1:] - train_mean)/train_std
    X_test[:,1:] = (X_test[:,1:] - train_mean)/train_std
    return X_train,X_test 

X, y = load_breast_cancer().data, load_breast_cancer().target
X = np.column_stack((np.array([1]*len(X)),X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_train, n_test = len(y_train),len(y_test)
X_train, X_test = normalize(X_train,X_test)

sig = lambda a: 1/(1+exp(-a))
loss = lambda X,Y,w: -sum([Y[i]*log(sig(w.T.dot(X[i])))+\
                           (1-Y[i])*log(1-sig(w.T.dot(X[i]))) for i in range(len(X))])
accuracy = lambda X,Y,w: 1 - sum(abs([1 if x >= 0 else 0 for x in X.dot(w)]-Y))/len(X)
      
w = np.random.uniform(low = -1,high = 1,size = X_train.shape[1])
n_epochs = 1
eta = 0.01
order = list(range(n_train))
train_acc = [];test_acc = []
train_error = [];test_error = []
for epoch in range(n_epochs):
    np.random.shuffle(order)
    for i in order:
        xi = X_train[i:i+1]
        yi = y_train[i:i+1] 
        gradients = xi.T.dot(sig(xi.dot(w)) - yi)
        w = w - eta * gradients
        train_acc.append(accuracy(X_train,y_train,w))
        test_acc.append(accuracy(X_test,y_test,w)) 
        train_error.append(loss(X_train,y_train,w)/n_train)
        test_error.append(loss(X_test,y_test,w)/n_test)
        
plt.plot(list(range(len(X_train))),train_acc,label = 'train accuracy')
plt.plot(list(range(len(X_train))),test_acc,label = 'test accuracy')
plt.xlabel('number of iterations');plt.ylabel('accuracy')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(list(range(len(X_train))),train_error,label = 'train error')
plt.plot(list(range(len(X_train))),test_error,label = 'test_error')
plt.xlabel('number of iterations');plt.ylabel('error')
plt.legend(loc = 'upper right')
plt.show()

print('The learned parameter vector w is ',w[1:])
print('The final train cross-entropy is ',train_error[-1]*n_train)
print('The final test cross-entropy is ',test_error[-1]*n_test)
print('The final train accuracy is ', train_acc[-1])
print('The final test accuracy is ', test_acc[-1])