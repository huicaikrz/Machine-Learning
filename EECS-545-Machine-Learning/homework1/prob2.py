import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

RMSE = lambda y,y_fit: ((y-y_fit)**2).mean()**0.5

def normalize():
    train_mean = X_train[:,1:].mean(axis=0);train_std = X_train[:,1:].std(axis=0)
    X_train[:,1:] = (X_train[:,1:] - train_mean)/train_std
    X_test[:,1:] = (X_test[:,1:] - train_mean)/train_std
# Load dataset
dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target
raw = np.array(features)
features = np.column_stack((np.array([1]*len(features)),features))

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
n_train = len(y_train)
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
n_test = len(y_test)

train_error = [];test_error = []
#order 0
X_train, X_test = np.array([1]*n_train), np.array([1]*n_test)
theta = 1/(X_train.T.dot(X_train))*(X_train.T).dot(y_train)
test_error.append(RMSE(y_test,X_test.dot(theta)))
train_error.append(RMSE(y_train,X_train.dot(theta)))

for i in range(1,5,1):
    if i == 1:
        new_features = np.array(features)
    else:
        new_features = np.column_stack((new_features,raw**i))
    X_train, X_test = new_features[:-Nsplit], new_features[-Nsplit:]
    normalize()
    theta = np.linalg.solve(a=X_train.T.dot(X_train),b = X_train.T.dot(y_train))
    train_error.append(RMSE(y_train,X_train.dot(theta)))
    test_error.append(RMSE(y_test,X_test.dot(theta)))
    
plt.plot(list(range(5)), test_error[:],label = 'test error')
plt.xlabel('order');plt.ylabel('error') 
plt.plot(list(range(5)), train_error[:],label = 'train error')
plt.legend(loc = 'upper left')
plt.show()

#b
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

train_error = [];test_error = [] 
for i in list(range(20,120,20)):
    n_train = round(len(X_train)*i/100)
    Xtrain, ytrain = X_train[:n_train], y_train[:n_train]
    theta = pinv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(ytrain)
    train_error.append(RMSE(ytrain,Xtrain.dot(theta)))
    test_error.append(RMSE(y_test,X_test.dot(theta)))
    
plt.plot([i/100 for i in range(20,120,20)], train_error, label = 'train error')
plt.xlabel('train set size');plt.ylabel('error')
plt.plot([i/100 for i in range(20,120,20)], test_error,label = 'test error')
plt.legend(loc = 'upper right')
plt.show()