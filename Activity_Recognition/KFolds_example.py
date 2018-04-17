import numpy as np
import Image_Processing as ip
from sklearn.model_selection import KFold # import KFold

train = ip.train_data[-50:]
#test = ip.test_data[-50:]

X = np.array([i[0] for i in train]).reshape(-1, ip.IMAGE_SIZE, ip.IMAGE_SIZE, 1)
y = [i[1] for i in train]
y = np.reshape(y, [-1, 2])

#test_x = np.array([i[0] for i in test]).reshape(-1, ip.IMAGE_SIZE, ip.IMAGE_SIZE, 1)
#test_y = [i[1] for i in test]
#test_y = np.reshape(test_y, [-1, 2])

#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6], [8, 7], [9, 2], [7, 6]]) # create an array
#y = np.array([1, 2, 3, 4, 5, 6, 7, 8]) # Create another array

n_folds = ip.train_data.size / 50
print(n_folds)
 

kf = KFold(n_splits=50) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 
KFold(n_splits=50, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]