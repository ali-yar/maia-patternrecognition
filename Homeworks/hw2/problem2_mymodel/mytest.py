import numpy as np
import tensorflow as tf
import tflearn
from tflearn import data_preprocessing as data_prep

# THE ONLY VARIABLES TO CHANGE
testFile = "./test_data.txt" # where to get the data
delimiter = " " # how is data separated
model = "mymodel.tfl" # where to get the trained weights

# configuration for network
hiddenNodes = 180
activ = 'relu'
weight = 'xavier'
optimizerFunc = 'sgd'

# load test data
test_set = np.genfromtxt(testFile, delimiter = delimiter)

# set label of positives to 1 and of negatives to 0 regardless of original labels
test_set[test_set[:,-1]>0,-1] = 1;
test_set[test_set[:,-1]<=0,-1] = 0;

testfv = test_set[:,:-1]
testlab = test_set[:,-1].flatten()

# input dimension
d = testfv.shape[1]



with tf.Graph().as_default():
    # set the data normalization scheme
    prep = data_prep.DataPreprocessing()
    prep.add_featurewise_zero_center()
    prep.add_featurewise_stdnorm()

    # Structure of the network
    net = tflearn.input_data([None, d], data_preprocessing=prep)
    net = tflearn.normalization.batch_normalization(net)                
    net = tflearn.fully_connected(net, hiddenNodes, activation=activ, weights_init=weight)                
    net = tflearn.normalization.batch_normalization(net)                
    net = tflearn.fully_connected(net, hiddenNodes, activation=activ, weights_init=weight)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    
    # apply regression
    net = tflearn.regression(net, optimizer=optimizerFunc, loss='categorical_crossentropy')
    
    lm = tflearn.DNN(net,tensorboard_verbose=0)
   
    lm.load(model)
    
 
print("Accuracy: {}%".format(100 * np.mean(testlab == np.argmax(lm.predict(testfv), axis=1))))