import numpy as np
from math import log

# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")

# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")

N_spam = sum(train_labels)
n_train = len(train_labels)

#prior
p_spam = (N_spam+1)/(n_train + 2*1)
p_nspam = (n_train-N_spam+1)/(n_train + 2*1)

#prob fearure is 1 in spam
p_spam_fea = (train_features[train_labels > 0].sum(axis = 0)+1)/(N_spam+2)
#prob fearure is 1 in but not spam
p_nspam_fea = (train_features[train_labels < 0.5].sum(axis = 0)+1)/(n_train - N_spam + 2)

pre_test = [1 if log(p_spam) + sum(np.log(p_spam_fea[test_features[i,:] > 0 ]))+sum(np.log(1-p_spam_fea[test_features[i,:] < 0.5 ]))>
                 log(p_nspam) + sum(np.log(p_nspam_fea[test_features[i,:] > 0 ]))+sum(np.log(1-p_nspam_fea[test_features[i,:] < 0.5 ]))
              else 0 for i in range(len(test_labels))]

error_rate = sum(abs(test_labels-pre_test))/len(test_labels)
print('The error rate on test data is',error_rate*100,'%') 