import os
import numpy as np
from sklearn import svm, preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

path = os.path.dirname(os.path.abspath("__file__"))

# options
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
nus = np.round(np.arange(0.01,1.0,0.01),2)
degree = 2 # applicable for 'poly' kernel only
n = 5 # number of cross validation folds

kernels = ['rbf']
nus = [0.35]

# create a parameter grid 
param_grid = dict(kernel=kernels, nu=nus)

# read the data set
dataset = np.genfromtxt(path + "/hw3data.csv", delimiter = ",")

# shuffle
#np.random.shuffle(dataset)
    
# separate features and labels
X = dataset[:,:-1]
Y = dataset[:,-1]

# standardize features
X = preprocessing.scale(X)

# build classifier
clf = svm.NuSVC(degree=degree)

# create a grid search for best params with cross validation 
grid = GridSearchCV(clf, param_grid, cv=n, scoring='roc_auc')

# fit grid search
grid.fit(X, Y)

# grid search results 
grid.grid_scores_

# check best params
print("Best score={} obtained with params={}".format(grid.best_score_,grid.best_params_))

# build classifier with best params
v = grid.best_params_['nu']
k = grid.best_params_['kernel']
clf = svm.NuSVC(nu=v, kernel=k, degree=degree)

# fit classifier
clf.fit(X, Y)

prefix = "p2-cv" + str(n) + "-auc" + str(grid.best_score_) + "-v" + str(v)

# save model
joblib.dump(clf, prefix + "-mymodel.pkl")

# save grid search results
joblib.dump(grid, prefix + "-gridsearch.pkl") 


roc_auc_score(Y,clf.predict(X))

# test on new data
#d = np.genfromtxt(path + "/test_data.txt", delimiter = " ")
#clf.fit(X[0:6000,:], Y[0:6000,])
## separate features and labels
#x = d[:,:-1]
#y = d[:,-1]
#x = preprocessing.scale(x)
#roc_auc_score(y,clf.predict(x))
