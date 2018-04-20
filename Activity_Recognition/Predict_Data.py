import Model_Activity_Recognition as mar
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#change these global variable if you want to predict using different model
IMAGE_SIZE = mar.IMAGE_SIZE
MODEL_NAME = mar.MODEL_NAME
FILE_PARAMETER = mar.FILE_PARAMETER

#PREDICT TRAINING AND TEST SET
#Gets train and test data
train = np.load('train_data.npy')
test = np.load('test_data.npy')

#Gives format to the data and the labels
x = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y = [i[1] for i in train]
y = np.reshape(y, [-1, 5])

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
test_y = [i[1] for i in test]
test_y = np.reshape(test_y, [-1, 5])

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    f = open(FILE_PARAMETER,'r')
    p = f.readline().split(',')
    
    print('Create model : weight = ', p[0], ', bias = ', p[1], ', learning rate = ', p[2])
    model = mar.create_model(int(p[0]), int(p[1]), float(p[2]), 3)
    print('Load model: ', MODEL_NAME)
    model.load(MODEL_NAME)
    
    #predict training set
    print('Predict training set...')
    y_pred = np.argmax(model.predict(x), 1)
    #print(y_pred)
    y_true = np.argmax(y, 1)
    #print(y_true)
    target_names = ['brushingteeth', 'cuttinginkitchen', 'jumpingjack', 'lunges', 'wallpushups']
    
    print('Performance evaluation in the training set:')
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('accuracy = ', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=target_names))

    
    #predict testing set
    print('Predict testing set...')
    test_pred = np.argmax(model.predict(test_x), 1)
    #print(test_pred)
    test_true = np.argmax(test_y, 1)
    #print(test_true)
    
    print('Performance evaluation in the testing set:')
    print('Confusion matrix:')
    print(confusion_matrix(test_true, test_pred))
    print('accuracy = ', accuracy_score(test_true, test_pred))
    print(classification_report(test_true, test_pred, target_names=target_names))
    
else:
    print('Model not found')




