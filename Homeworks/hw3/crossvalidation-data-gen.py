#import os
#import numpy as np
#from sklearn import preprocessing
#
#path = os.path.dirname(os.path.abspath("__file__")) + "/"
#
#n = 5
#for i in range(n) :          
#    # Read the data set
#    dataset = np.genfromtxt(path + "hw3data.csv", delimiter = ",")
#    
#    # standardize features
#    dataset[:,:-1] = preprocessing.scale(dataset[:,:-1])
#    
#    # shuffle dataset
##    np.random.shuffle(dataset)
#    
#    # build the positive and negative subsets
#    pos = np.copy(dataset[dataset[:,-1]>0,:])
#    neg = np.copy(dataset[dataset[:,-1]<=0,:])
#    
#    pos_split = np.split(pos,n);
#    neg_split = np.split(neg,n);
#    
#    train_pos_set = np.concatenate((pos_split[(i+1)%n], pos_split[(i+2)%n]), axis=0)
#    valid_pos_set = np.concatenate((pos_split[(i+3)%n], pos_split[(i+4)%n]), axis=0)
#    
#    train_neg_set = np.concatenate((neg_split[(i+1)%n], neg_split[(i+2)%n]), axis=0)
#    valid_neg_set = np.concatenate((neg_split[(i+3)%n], neg_split[(i+4)%n]), axis=0)
#    
#    test_pos_set = pos_split[i]
#    test_neg_set = neg_split[i]
#    
#    
#    train_set = np.concatenate((train_pos_set, train_neg_set), axis=0)
#    valid_set = np.concatenate((valid_pos_set, valid_neg_set), axis=0)
#    test_set = np.concatenate((test_pos_set, test_neg_set), axis=0)
#    
#    # save into files
#    folder = "data/"
#    np.savetxt(path+folder+"train_"+ str(i+1) +".txt",train_set, delimiter=',')
#    np.savetxt(path+folder+"valid_"+ str(i+1) +".txt",valid_set, delimiter=',')
#    np.savetxt(path+folder+"test_"+ str(i+1) +".txt",test_set, delimiter=',')
#    


# # # use this for 10-fold cross validation # # # 

#import os
#import numpy as np
#from sklearn import preprocessing
#
#path = os.path.dirname(os.path.abspath("__file__")) + "/"
#
#n = 10
#for i in range(n) :          
#    # Read the data set
#    dataset = np.genfromtxt(path + "hw3data.csv", delimiter = ",")
#    
#    # standardize features
#    dataset[:,:-1] = preprocessing.scale(dataset[:,:-1])
#    
#    # shuffle dataset
##    np.random.shuffle(dataset)
#    
#    # build the positive and negative subsets
#    pos = np.copy(dataset[dataset[:,-1]>0,:])
#    neg = np.copy(dataset[dataset[:,-1]<=0,:])
#    
#    pos_split = np.split(pos,n);
#    neg_split = np.split(neg,n);
#    
#    train_pos_set = np.concatenate(( pos_split[(i+1)%n], pos_split[(i+2)%n],
#                                     pos_split[(i+3)%n], pos_split[(i+4)%n],
#                                     pos_split[(i+5)%n], pos_split[(i+6)%n],
#                                     pos_split[(i+7)%n] ), axis=0)
#    valid_pos_set = np.concatenate((pos_split[(i+8)%n], pos_split[(i+9)%n]), axis=0)
#    test_pos_set = pos_split[i]
#    
#    
#    train_neg_set = np.concatenate(( neg_split[(i+1)%n], neg_split[(i+2)%n],
#                                     neg_split[(i+3)%n], neg_split[(i+4)%n],
#                                     neg_split[(i+5)%n], neg_split[(i+6)%n],
#                                     neg_split[(i+7)%n] ), axis=0)
#    valid_neg_set = np.concatenate((neg_split[(i+8)%n], neg_split[(i+9)%n]), axis=0)
#    test_neg_set = neg_split[i]
#    
#    
#    train_set = np.concatenate((train_pos_set, train_neg_set), axis=0)
#    valid_set = np.concatenate((valid_pos_set, valid_neg_set), axis=0)
#    test_set = np.concatenate((test_pos_set, test_neg_set), axis=0)
#    
#    # save into files
#    folder = "data/"
#    np.savetxt(path+folder+"cv10_train_"+ str(i+1) +".txt",train_set, delimiter=',')
#    np.savetxt(path+folder+"cv10_valid_"+ str(i+1) +".txt",valid_set, delimiter=',')
#    np.savetxt(path+folder+"cv10_test_"+ str(i+1) +".txt",test_set, delimiter=',')
#    