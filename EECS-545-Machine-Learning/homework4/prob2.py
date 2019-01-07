import numpy as np

from sklearn import datasets, svm
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='./')
#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

N = len(images)
np.random.seed(1234)
inds = np.random.permutation(N)
images = np.array([images[i] for i in inds])
targets = np.array([targets[i] for i in inds])

# Normalize data
X_data = images/255.0
Y = targets

# Train/test split
X_train, y_train = X_data[:10000], Y[:10000]
X_test, y_test = X_data[-10000:], Y[-10000:]

#b
clf = svm.SVC(kernel='rbf',C = 1,gamma=1)
clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,clf.predict(X_test))
print('The accuracy of the train model on the test set is ', accuracy)
#c
#when gamma is large, all data points will have roughly the same weight,
#because of the exp, then it will cause high bias but low variance
#when gamma is small, cetain data point will have more weight
#high variance, low bias

#d
from sklearn.model_selection import GridSearchCV
parameters = {'C':[1,3,5], 'gamma':[0.05,0.1,0.5,1.0]}
svc = svm.SVC(kernel='rbf')
clf = GridSearchCV(svc, parameters,cv = 5,n_jobs = -1)  
clf.fit(X_train,y_train)  
accuracy_score(y_test,clf.predict(X_test))


