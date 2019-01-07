import numpy as np
from sklearn import datasets

RMSE = lambda y,y_fit: ((y-y_fit)**2).mean()**0.5

dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target
features = np.column_stack((np.array([1]*len(features)),features))

Nsplit = 50;n_val = round((len(labels)-Nsplit)*0.1)
# Training set
X_train, y_train = features[:-Nsplit-n_val], labels[:-Nsplit-n_val]
n_train = len(y_train)
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
X_val, y_val = features[-Nsplit-n_val:-Nsplit], labels[-Nsplit-n_val:-Nsplit]

train_mean = X_train[:,1:].mean(axis=0);train_std = X_train[:,1:].std(axis=0)
X_train[:,1:] = (X_train[:,1:] - train_mean)/train_std
X_test[:,1:] = (X_test[:,1:] - train_mean)/train_std
X_val[:,1:] = (X_val[:,1:] - train_mean)/train_std

train_error = [];test_error = [];val_error = []
for lamda in range(6):
    lamda /= 10
    theta = np.linalg.inv(X_train.T.dot(X_train) + n_train*lamda*
                          np.diag([1]*X_train.shape[1])).dot(X_train.T).dot(y_train)
    test_error.append(RMSE(y_test,X_test.dot(theta)))
    train_error.append(RMSE(y_train,X_train.dot(theta)))
    val_error.append(RMSE(y_val,X_val.dot(theta)))
    
best_lamda = val_error.index(min(val_error))/10
print('Best lamda is: ',best_lamda)
print('The validation error for this lamda is: ', min(val_error))
print('The test error for this lamda is: ', test_error[int(best_lamda*10)])