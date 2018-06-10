import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from maketerraindata import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = SVC(kernel="linear",C=10)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)

X = np.asarray(features_train)
Y = np.asarray(labels_train)

plt.clf()
plt.scatter(X[:,0], X[:, 1], c=labels_train, s=10, cmap=plt.cm.Paired)


# plot the decision function
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

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
