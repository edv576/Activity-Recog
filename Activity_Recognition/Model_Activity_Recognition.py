import tflearn
import os
import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import KFold


IMAGE_SIZE = 120
MODEL_NAME = 'activity_recognition-{}-{}.model'.format('final','6conv-basic-video')
MODEL_NAME_TEMPORAL = 'activity_recognition_temporal-{}-{}.model'.format('final','6conv-basic-video')
FILE_PARAMETER = 'parameter.csv' 

def create_model(weight, bias, learning_rate, depth): #32, 2
    tf.reset_default_graph()
    #Format of the input data for the model
    convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, depth], name='input')
    
    #Layers of the model
    convnet = conv_2d(convnet, bias, weight, activation='relu')
    convnet = max_pool_2d(convnet, weight)
    
    convnet = conv_2d(convnet, 2*bias, weight, activation='relu')
    convnet = max_pool_2d(convnet, weight)
    
    convnet = conv_2d(convnet, bias, weight, activation='relu')
    convnet = max_pool_2d(convnet, weight)
    
    convnet = conv_2d(convnet, 2*bias, weight, activation='relu')
    convnet = max_pool_2d(convnet, weight)
    
    convnet = conv_2d(convnet, bias, weight, activation='relu')
    convnet = max_pool_2d(convnet, weight)
    
    convnet = conv_2d(convnet, 2*bias, weight, activation='relu')
    convnet = max_pool_2d(convnet, weight)
    
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 5, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='target')
    
    #Creates the model using the configuration created
    model = tflearn.DNN(convnet, tensorboard_dir='log')
      
    return model

    
def cross_validate(train_x_all, train_y_all, split_size=3):
    print ('Running cross-validation...')
    bestParameter = []
    weights = [2, 2, 4, 4, 2, 2, 4, 4]
    biases = [24, 32, 24, 32, 24, 32, 24, 32]
    learning_rates = [0.1, 0.1, 0.1, 0.1, 0.001, 0.001, 0.001, 0.001]
    parameters = zip(weights, biases, learning_rates)
    kf = KFold(n_splits=split_size)
    bestAvgAccuracy = 0
    #Iterates over the parameter (weight and bias)
    p = 1
    for weight, bias, learning_rate in parameters:
        print('Process parameter ', p, ': weight = ', weight, ', bias = ', bias, ', learning rate = ', learning_rate)
        model = create_model(weight, bias, learning_rate, 3)

        #Iterates over the fold configuration
        c = 1
        sumAccuracy = 0
        for train_idx, val_idx in kf.split(train_x_all, train_y_all):
            print('Process fold ', c, '...')
            train_x = train_x_all[train_idx]
            train_y = train_y_all[train_idx]
            val_x = train_x_all[val_idx]
            val_y = train_y_all[val_idx]
            
            #Trains model using one of the fold configuration
            model.fit(train_x, train_y, n_epoch=5, validation_set=({'input':val_x}, {'target':val_y}), 
                      snapshot_step=None, snapshot_epoch=False, show_metric=False, run_id='parameter_model'+ str(p))
            
            #Add results of accuracy to array
            Acc = model.evaluate(val_x, val_y)
            print('Process fold ', c, ': Accuracy = ', Acc[0])
            sumAccuracy = sumAccuracy + Acc[0]
            c = c + 1
        
        avgAccuracy = sumAccuracy/split_size
        print('Process parameter ', p, ': Average accuracy = ', avgAccuracy)
        if (avgAccuracy > bestAvgAccuracy) :
            bestAvgAccuracy = avgAccuracy
            bestParameter = [weight, bias, learning_rate]
            
        p = p + 1
        
    print('Best parameter (Accuracy = ',bestAvgAccuracy,') : weight = ', bestParameter[0], 
    ', bias = ', bestParameter[1], ', learning rate = ', bestParameter[2])
    return bestParameter

    
#--RUNNING CROSS-VALIDATION AND CREATE THE MODEL--
if not os.path.exists('{}.meta'.format(MODEL_NAME)):
    #Gets train data
    train = np.load('train_data.npy')
    
    #Gives format to the data and the labels
    x = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    y = [i[1] for i in train]
    y = np.reshape(y, [-1, 5])
    print(y)
        
    #Performs cross-validation
    bestParameterCV = cross_validate(x, y)
    
    print ('Saving the best parameter...')
    f = open(FILE_PARAMETER,'w')
    f.write(str(bestParameterCV[0]) + ',' + str(bestParameterCV[1]) + ',' + str(bestParameterCV[2]))
    f.close()
      
    print ('Creating the spatial model...')
    final_model = create_model(bestParameterCV[0], bestParameterCV[1], bestParameterCV[2], 3)
    final_model.fit(x, y)
      
    #Saves the model to file
    print ('Saving the spatial model as ', MODEL_NAME)
    final_model.save(MODEL_NAME)
    
#    print ('Creating the temporal model...')
#    final_model = create_model(bestParameterCV[0], bestParameterCV[1], bestParameterCV[2], 20)
#    final_model.fit(x, y)
#      
#    #Saves the model to file
#    print ('Saving the temporal model as ', MODEL_NAME_TEMPORAL)
#    final_model.save(MODEL_NAME_TEMPORAL)



