import numpy as np

n = 5
    
for i in range(n) :          
    # Read the data set
    dataset = np.genfromtxt("./data/hw2data.csv", delimiter = ",")
    
    # build the positive and negative subsets
    pos = np.copy(dataset[dataset[:,-1]>0,:])
    neg = np.copy(dataset[dataset[:,-1]<=0,:])
    
    # Make training set (75%) and test set (25%)
    pos_split = np.split(pos,n);
    neg_split = np.split(neg,n);
    
    train_pos_set = np.concatenate((pos_split[(i+1)%n], pos_split[(i+2)%n], pos_split[(i+3)%n], pos_split[(i+4)%n]), axis=0)
    train_neg_set = np.concatenate((neg_split[(i+1)%n], neg_split[(i+2)%n], neg_split[(i+3)%n], neg_split[(i+4)%n]), axis=0)
    
    test_pos_set = pos_split[i]
    test_neg_set = neg_split[i]
    
    
    train_set = np.concatenate((train_pos_set, train_neg_set), axis=0)
    test_set = np.concatenate((test_pos_set, test_neg_set), axis=0)
    
    # shuffle training set
    np.random.shuffle(train_set)
    np.random.shuffle(train_set)
    np.random.shuffle(train_set)
    
    path = "./data/"
    np.savetxt(path+"train_"+ str(i+1) +".txt",train_set, delimiter=',')
    np.savetxt(path+"test_"+ str(i+1) +".txt",test_set, delimiter=',')
    