import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold

#iris = datasets.load_iris()
#
#X_train, X_test, y_train, y_test = train_test_split(
#                         iris.data, iris.target, test_size=0.4, random_state=0)
#
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)                         

kf = KFold(25, n_folds=5, shuffle=False)


for i, data in enumerate(kf):
    print(data[0])
