import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from makeData import getCVData

# number of cross validation folds
n = 5

# params
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
nus = np.round(np.arange(0.01,1.01,0.01),2) # create range from 0.01 to 1.00 with 0.01 step 

for kernel in kernels:
    AUC = np.zeros(n)
    print("\n\nKernel: " + kernel)
    for i in range(n):
        bestauc = -1
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test= getCVData(i+1)
        for v in nus:
            # instanstiate classifier
            clf = svm.NuSVC(nu=v, kernel=kernel, degree=2)
            # train classifier
            clf.fit(X_train, Y_train)
            # predict
            Y_predict = clf.predict(X_valid)
            # measure score
            auc = roc_auc_score(Y_valid, Y_predict)
            # update best params and model
            if auc > bestauc:
                bestauc = auc
                bestv = v
                bestSVM = clf
        # predict with best model                
        Y_predict = bestSVM.predict(X_test)
        # measure score
        auc = roc_auc_score(Y_test, Y_predict)
        AUC[i] = auc
        print("cv = {}  -  best v = {}  - best auc = {}".format(str(i+1),bestv,bestauc))
    print("Average: mean = {}  -  std = {}".format(AUC.mean(),AUC.std()))