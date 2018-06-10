import os
import numpy as np

# p1
def getCVData(i):
    path = os.path.dirname(os.path.abspath("__file__")) + "/"
    pathData = path + "data/"

    fTrain = "train_" + str(i) + ".txt"
    fValid = "valid_" + str(i) + ".txt"
    fTest = "test_" + str(i) + ".txt"
    
    # load files
    train_set = np.genfromtxt(pathData+fTrain, delimiter = ",")
    valid_set = np.genfromtxt(pathData+fValid, delimiter = ",")
    test_set = np.genfromtxt(pathData+fTest, delimiter = ",")
    
    # separate features and labels
    X_train = train_set[:,:-1]
    Y_train = train_set[:,-1]
    X_valid = valid_set[:,:-1]
    Y_valid = valid_set[:,-1]
    X_test = test_set[:,:-1]
    Y_test = test_set[:,-1]
    
#    # standardize data
#    scaler = preprocessing.StandardScaler().fit(X_train)                              
#    X_train = scaler.transform(X_train)
#    X_valid = scaler.transform(X_valid)  
#    X_test = scaler.transform(X_test)  
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# p2
def getCV5Data2(i):
    path = os.path.dirname(os.path.abspath("__file__")) + "/"
    pathData = path + "data/"

    fTrain = "train_" + str(i) + ".txt"
    fValid = "valid_" + str(i) + ".txt"
    fTest = "test_" + str(i) + ".txt"
    
    # load files
    train_set = np.genfromtxt(pathData+fTrain, delimiter = ",")
    valid_set = np.genfromtxt(pathData+fValid, delimiter = ",")
    test_set = np.genfromtxt(pathData+fTest, delimiter = ",")
    
    # combine train and validate sets into 1
    train_set = np.vstack((train_set,valid_set))
    
    # separate features and labels
    X_train = train_set[:,:-1]
    Y_train = train_set[:,-1]
    X_test = test_set[:,:-1]
    Y_test = test_set[:,-1]
    
    return X_train, Y_train, X_test, Y_test


# p2
def getCV10Data2(i):
    path = os.path.dirname(os.path.abspath("__file__")) + "/"
    pathData = path + "data/"

    fTrain = "cv10_train_" + str(i) + ".txt"
    fValid = "cv10_valid_" + str(i) + ".txt"
    fTest = "cv10_test_" + str(i) + ".txt"
    
    # load files
    train_set = np.genfromtxt(pathData+fTrain, delimiter = ",")
    valid_set = np.genfromtxt(pathData+fValid, delimiter = ",")
    test_set = np.genfromtxt(pathData+fTest, delimiter = ",")
    
    # combine train and validate sets into 1
    train_set = np.vstack((train_set,valid_set))
    
    # separate features and labels
    X_train = train_set[:,:-1]
    Y_train = train_set[:,-1]
    X_test = test_set[:,:-1]
    Y_test = test_set[:,-1]  
    
    return X_train, Y_train, X_test, Y_test



# non used
def getTrainTestData2(i):
    path = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Pattern Recognition/Homeworks/Hw3/"
    pathData = path + "data/"

    fTrain = "p2-train_" + str(i) + ".txt"
    fTest = "p2-test_" + str(i) + ".txt"
    
    train_set = np.genfromtxt(pathData+fTrain, delimiter = ",")
    test_set = np.genfromtxt(pathData+fTest, delimiter = ",")
    
    # separate features and labels
    X_train = train_set[:,:-1]
    Y_train = train_set[:,-1]
    X_test = test_set[:,:-1]
    Y_test = test_set[:,-1]

    return X_train, Y_train, X_test, Y_test