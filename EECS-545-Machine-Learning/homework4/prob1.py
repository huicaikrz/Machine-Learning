import numpy as np 
from sklearn import svm
import matplotlib.pyplot as plt 

np.random.seed(3)

mean_1 = [ 2.0 , 0.2 ]
cov_1 = [ [ 1 , .5 ] , [ .5 , 2.0 ]]

mean_2 = [ 0.4 , -2.0 ]
cov_2 = [ [ 1.25 , -0.2 ] , [ -0.2, 1.75 ] ]

x_1 , y_1 = np.random.multivariate_normal( mean_1 , cov_1, 15).T
x_2 , y_2 = np.random.multivariate_normal( mean_2 , cov_2, 15).T

X = np.zeros((30,2))
X[0:15,0] = x_1
X[0:15,1] = y_1
X[15:,0] = x_2
X[15:,1] = y_2

y = np.zeros(30)
y[0:15] = np.ones(15)
y[15:] = -1 * np.ones(15)

def plot_dataset(X, y, axes):
    plt.plot(X[0:15,0], X[0:15,1] , 'x' )
    plt.plot(X[15:,0], X[15:,1] , 'ro')
    plt.axis(axes)
    
def plot_svc_decision_boundary(svm_clf, xmin, xmax,c):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    #svs = svm_clf.support_vectors_
    #plt.scatter(svs[:, 0], svs[:, 1], s=20, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    plt.title('Identity C=%s' %str(C))
    
def plot_predictions(clf, axes,C):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    plt.title('Gaussian C=%s' %str(C))
    
for C in [1,100]:
    clf = svm.SVC(kernel='linear',C = C)
    clf.fit(X,y)
    print(len(clf.support_vectors_))
    plot_dataset(X,y,[-4,6,-6,3])
    plot_svc_decision_boundary(clf,-6,5,C)
    plt.show()
    
for C in [1,3]:
    clf = svm.SVC(kernel='rbf',C = C)
    clf.fit(X,y)
    print(len(clf.support_vectors_))
    plot_dataset(X,y,[-4,6,-6,3])
    plot_predictions(clf,[-4,6,-6,3],C)
    plt.show()
    



