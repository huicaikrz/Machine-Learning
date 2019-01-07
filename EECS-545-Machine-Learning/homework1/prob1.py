import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

MSE = lambda y,y_fit: ((y-y_fit)**2).mean()

def normalize():
    train_mean = X_train[:,1:].mean(axis=0);train_std = X_train[:,1:].std(axis=0)
    X_train[:,1:] = (X_train[:,1:] - train_mean)/train_std
    X_test[:,1:] = (X_test[:,1:] - train_mean)/train_std

def result_show(qnumber):
    if qnumber in ['b','c']:
        plt.plot(list(range(n_epochs)), train_error)
        plt.xlabel('number of epochs');plt.ylabel('train error')
        plt.show()
    print('Learned weight vector: ', theta[1:])
    print('The bias term: ', theta[0])
    print('Train error: ', train_error[-1])
    print('Test error: ', test_error[-1])

# Load dataset
dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target
features = np.column_stack((np.array([1]*len(features)),features))

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
n_train = len(y_train)
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
n_test = len(y_test)

normalize()   
# SGD
n_epochs = 500
theta = np.random.uniform(low = -0.1,high = 0.1,size = features.shape[1])
order = list(range(n_train))
eta = 5*10**(-3) #rate
train_error = [];test_error = []
for epoch in range(n_epochs):
    np.random.shuffle(order)
    for i in order:
        xi = X_train[i:i+1]
        yi = y_train[i:i+1] 
        gradients = 2/n_train*xi.T.dot(xi.dot(theta) - yi)
        theta = theta - eta * gradients
    train_error.append(MSE(y_train,X_train.dot(theta)))
    test_error.append(MSE(y_test,X_test.dot(theta))) 
result_show('b')

# BGD
n_epochs = 500
theta = np.random.uniform(low = -0.1,high = 0.1,size = features.shape[1])
eta = 5*10**(-2)
train_error = [];test_error = []
for epoch in range(n_epochs):
    gradients = 2/n_train*X_train.T.dot(X_train.dot(theta) - y_train)
    theta = theta - eta * gradients
    train_error.append(MSE(y_train,X_train.dot(theta)))
    test_error.append(MSE(y_test,X_test.dot(theta)))    
result_show('c')

#closed form
theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
train_error = [MSE(y_train,X_train.dot(theta))]
test_error = [MSE(y_test,X_test.dot(theta))]               
result_show('d')

#e
features_orig = features
labels_orig = labels
train_error = [];test_error = []
for k in range(100):
  # Shuffle data
    rand_perm = np.random.permutation(len(features))
    features = np.array([features_orig[ind] for ind in rand_perm])
    labels = np.array([labels_orig[ind] for ind in rand_perm])
    # Train/test split
    Nsplit = 50
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    normalize()  
    theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    train_error.append(MSE(y_train,X_train.dot(theta)))
    test_error.append(MSE(y_test,X_test.dot(theta)))                
print('Mean training error: ', np.mean(train_error))
print('Mean test error: ', np.mean(test_error))