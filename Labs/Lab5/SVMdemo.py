"""
Plot the maximum margin separating hyperplane within a two-class
dataset using a Support Vector Machine classifier with
various kernels.
"""


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from sklearn import svm



# create a double banana-shaped dataset 

N = np.array([500, 500])
r = 1
s = 0.1

domaina = 0.125*pi + np.random.rand(N[0])*1.25*pi
domainb = 0.375*pi - np.random.rand(1,N[1])*1.25*pi

X0 = np.zeros([N[0],2])
X1 = np.zeros([N[1],2])

X0[:,0] = np.sin(domaina)
X0[:,1] = np.cos(domaina)
X0 = X0 + np.random.randn(N[0],2)*s

X1[:,0] = np.sin(domainb)
X1[:,1] = np.cos(domainb)
X1 = X1 + np.random.randn(N[1],2)*s+np.ones([N[1],2])*-0.75*r

X = np.vstack((X0, X1))

y = np.hstack((np.zeros(N[0]),np.ones(N[1])))

# fit the model
for fig_num, kernel in enumerate(('linear', 'poly', 'rbf')):
    clf = svm.SVC(kernel=kernel,C=10)
    clf.fit(X, y)
    
    plt.figure(fig_num)
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
    # plot support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50, facecolors='none',
                zorder=10, edgecolor='k')
    plt.title(kernel)
plt.show()
