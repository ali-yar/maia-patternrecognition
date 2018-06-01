import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from tflearn import data_preprocessing as data_prep
from tflearn.data_utils import to_categorical

# Read the data set
dataset = np.genfromtxt("C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Pattern Recognition/Lab/Lab4/pima-indians-diabetes.data", delimiter = ",")

# shuffle the samples
np.random.shuffle(dataset)

# number of features
d = dataset.shape[1] - 1

# Build the positive and negative subsets
pos = np.copy(dataset[dataset[:,-1]>0,:])
neg = np.copy(dataset[dataset[:,-1]==0,:])

# Make training and test set
pos_split = np.split(pos,4);
neg_split = np.split(neg,4);

train_pos_set = np.concatenate((pos_split[0], pos_split[1], pos_split[2]), axis=0)
train_neg_set = np.concatenate((neg_split[0], neg_split[1], neg_split[2]), axis=0)

test_pos_set = pos_split[3]
test_neg_set = neg_split[3]

train_set = np.concatenate((train_pos_set, train_neg_set), axis=0)
test_set = np.concatenate((test_pos_set, test_neg_set), axis=0)

X_train = train_set[:,:-1]
Y_train = to_categorical(train_set[:,-1:].flatten(),2)

X_test = test_set[:,:-1]
Y_test = to_categorical(test_set[:,-1:].flatten(),2)

# train a linear classifier
with tf.Graph().as_default():
    prep = data_prep.DataPreprocessing()
    prep.add_featurewise_zero_center()
    prep.add_featurewise_stdnorm()

    net = tflearn.input_data([None, d])
    net = tflearn.fully_connected(net, 150, activation='relu')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    gd = tflearn.Adam(learning_rate=0.01)
    net = tflearn.regression(net, optimizer=gd, loss='categorical_crossentropy')
    
    model = tflearn.DNN(net,tensorboard_verbose=0)
    model.fit(X_train, Y_train, n_epoch=500, show_metric=True)
    
print(np.argmax(model.predict(X_test), axis=1))
print("Accuracy: {}%".format(100 * np.mean(test_set[:,-1:].flatten() == np.argmax(model.predict(X_test), axis=1))))
