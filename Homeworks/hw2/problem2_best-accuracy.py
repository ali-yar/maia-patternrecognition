import os
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import tensorflow as tf
import tflearn
from tflearn import data_preprocessing as data_prep
from tflearn.data_utils import to_categorical

# n-fold crossvalidation / number of different splitting of data
n = 5 

# settings
optimizerFunc = 'sgd'
learnRate = 0.01

hiddenNodes = 180
epochs = 50
batchSize = 36

# options for the different combinations
weights = ['zeros', 'uniform', 'xavier']
batchNorm = [True, False]
activations = ['relu', 'tanh']

# where to find the train and test data files
pathData = "./data/"

# where to save the results (accuracies, models and AUC plots)
pathResult = "./result_{}_{}rate_{}nodes_{}epochs_{}batch/".format(optimizerFunc,learnRate,hiddenNodes,epochs,batchSize)

# create dir for results
if not os.path.exists(pathResult):
    os.makedirs(pathResult)

for i in range(n):
    expID = 0;
    for weight in weights:
        for isBatchNorm in batchNorm:
            for activ in activations:
                expID = expID + 1;
                batchType = 'batchnorm' if isBatchNorm else 'nobatchnorm'
                # train a MLP
                with tf.Graph().as_default():
                    # preparing data
                    # read from files
                    fTrain = "train_" + str(i+1) + ".txt"
                    fTest = "test_" + str(i+1) + ".txt"
                    train_set = np.genfromtxt(pathData+fTrain, delimiter = ",")
                    test_set = np.genfromtxt(pathData+fTest, delimiter = ",")
                    # set label of positives to 1 and of negatives to 0
                    train_set[train_set[:,-1]>0,-1] = 1;
                    train_set[train_set[:,-1]<=0,-1] = 0;
                    test_set[test_set[:,-1]>0,-1] = 1;
                    test_set[test_set[:,-1]<=0,-1] = 0;
                    # split into features and labels
                    X_train = train_set[:,:-1]
                    Y_train = to_categorical(train_set[:,-1].flatten(),2)
                    X_test = test_set[:,:-1]
                    Y_test = test_set[:,-1].flatten()
                    
                    # input dimension
                    d = X_train.shape[1]
                    
                    # set the data preprocessing scheme
                    prep = data_prep.DataPreprocessing()
                    prep.add_featurewise_zero_center()
                    prep.add_featurewise_stdnorm()
                    
                    # build the network configuration
                    net = tflearn.input_data([None, d], data_preprocessing=prep)
                    if isBatchNorm : net = tflearn.normalization.batch_normalization(net)                
                    net = tflearn.fully_connected(net, hiddenNodes, activation=activ, weights_init=weight)                
                    if isBatchNorm : net = tflearn.normalization.batch_normalization(net)                
                    net = tflearn.fully_connected(net, hiddenNodes, activation=activ, weights_init=weight)
                    net = tflearn.fully_connected(net, 2, activation='softmax')
                    
                    # apply regression
                    net = tflearn.regression(net, optimizer=optimizerFunc, learning_rate=learnRate, loss='categorical_crossentropy')
                    
                    # build the model
                    model = tflearn.DNN(net,tensorboard_verbose=0)
                    
                    # train the classifier
                    model.fit(X_train, Y_train, n_epoch=epochs, show_metric=True, batch_size=batchSize)
                    
                    # save model
                    modelName = "crossdata#{}--model#{}--{}_{}_{}.tfl".format(i+1,expID,weight,batchType,activ)
                    model.save(pathResult + modelName)
                
                # test the classifier
                predictions = model.predict(X_test)
                accuracy = 100 * np.mean( Y_test == np.argmax(predictions, axis=1) )
                
                print("Accuracy: {}%".format(accuracy))
                
                skplt.metrics.plot_roc(Y_test, predictions)
                plt.savefig(pathResult + "crossdata#{}--plot#{}--{}_{}_{}.jpg".format(i+1,expID,weight,batchType,activ))
                            
                with open(pathResult + "accuracy.txt","a") as f:
                    f.write("crossdata#{}--experiment#{}--{}_{}_{} = {}\n\n".format(i+1,expID,weight,batchType,activ,accuracy))
                
                tf.reset_default_graph() 
   
    