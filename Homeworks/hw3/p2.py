import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from makeData import getCV5Data2, getCV10Data2

# out-of-sample data  self generated to having rough final prediction of model
from sklearn import preprocessing
path = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Pattern Recognition/Homeworks/Hw3/"
d = np.genfromtxt(path + "test_data.txt", delimiter = " ")
x_new = d[:,:-1]
y_new = d[:,-1]
x_new = preprocessing.scale(x_new)


# number of cross validation folds
n = 5

# params
kernels = ['rbf']
nus = np.round(np.arange(0.01,1.01,0.01),2) # create range from 0.01 to 1.00 with 0.01 step 

#nus = np.round(np.arange(0.35,0.60,0.01),2)
for kernel in kernels:
    AUC = np.zeros(n)
    print("\n\nKernel: " + kernel)
    for i in range(n):
        bestauc = -1
        X_train, Y_train, X_test, Y_test= getCV5Data2(i+1)
        for v in nus:
            # instanstiate classifier
            clf = svm.NuSVC(nu=v, kernel=kernel, degree=2)
            # train classifier
            clf.fit(X_train, Y_train)
            # predict
            Y_predict = clf.predict(X_test)
            # measure score
            auc = roc_auc_score(Y_test, Y_predict)
            # update best params and model
            if auc > bestauc:
                bestauc = np.round(auc,4)
                bestv = v
                bestSVM = clf
        # predict with best model  on new data              
        Y_predict = bestSVM.predict(x_new)
        auc = roc_auc_score(y_new, Y_predict)
        auc = np.round(auc,4)
        AUC[i] = auc
        
        Y_predict = bestSVM.predict(np.vstack((X_train,X_test)))
        wauc = roc_auc_score(np.concatenate((Y_train,Y_test), axis=0), Y_predict)
        wauc = np.round(wauc,4)
        print("cv={} - best_v={} - auc={} - whole_data_auc={} - new_data_auc={}".format(str(i+1),bestv,bestauc,wauc,auc))
        
    print("Average on new data: mean = {}  -  std = {}".format(AUC.mean(),AUC.std()))


# replace these 3 vars with best values found from results of above experiment
k = 'rbf'
v = 0.38
cv = 3

# build best model
X_train, Y_train, X_test, Y_test= getCV5Data2(cv)
clf = svm.NuSVC(nu=v, kernel=k, degree=2)
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
auc = roc_auc_score(Y_test, Y_predict)

# predict on new data              
Y_predict = clf.predict(x_new)
auc = roc_auc_score(y_new, Y_predict)
auc = np.round(auc,4)

# save model
joblib.dump(clf, "cv" + str(n) +  "-mymodel.pkl")

# fit on whole data and test on new data
X_whole = np.vstack((X_train,X_test))
Y_whole = np.concatenate((Y_train,Y_test), axis=0)
clf.fit(X_whole, Y_whole)
Y_predict = clf.predict(x_new)
auc = roc_auc_score(y_new, Y_predict)
auc = np.round(auc,4)



# test on new data
#from sklearn import preprocessing
#path = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Pattern Recognition/Homeworks/Hw3/"
#d = np.genfromtxt(path + "test_data.txt", delimiter = " ")
#clf = svm.NuSVC(nu=0.4, kernel='rbf', degree=2)
#clf.fit(X_train, Y_train)
#x = d[:,:-1]
#y = d[:,-1]
#x = preprocessing.scale(x)
#roc_auc_score(y,clf.predict(x))
