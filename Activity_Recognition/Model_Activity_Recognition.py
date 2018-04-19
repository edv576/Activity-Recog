import tflearn
import os
import numpy as np
import tensorflow as tf
import Image_Processing as ip
import Cross_validation as cval
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import KFold

tf.reset_default_graph()

#Format of the input data for the model
convnet = input_data(shape=[None, ip.IMAGE_SIZE, ip.IMAGE_SIZE, 3], name='input')

#Layers of the model
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=ip.LR, loss='categorical_crossentropy', name='target')

#Creates the model using the configuration created
model = tflearn.DNN(convnet, tensorboard_dir='log')

#model.fit({'input':x}, {'targets':y}, n_epoch=10, validation_set=({'input':x_test}, {'targets':y_test}), snapshot_step=500,
#          show_metric=True, run_id="mnist")

#model.save('tflearncnn.model')
  
#model.load('tflearncnn.model')

#print(model.predict([x_test[1]]))

#Loads the model if it was created previously
#if os.path.exists('{}.meta'.format(ip.MODEL_NAME)):
#    model.load(ip.MODEL_NAME)
#    print('Model loaded')

#Gets train and test data
train = ip.train_data
#test = ip.test_data

#Gives format to the data and the labels
x = np.array([i[0] for i in train]).reshape(-1, ip.IMAGE_SIZE, ip.IMAGE_SIZE, 3)
y = [i[1] for i in train]
y = np.reshape(y, [-1, 5])
print(y)

#test_x = np.array([i[0] for i in test]).reshape(-1, ip.IMAGE_SIZE, ip.IMAGE_SIZE, 1)
#test_y = [i[1] for i in test]
#test_y = np.reshape(test_y, [-1, 5])

#model.fit({'input':x}, {'targets':y}, n_epoch=5, validation_set=({'input':test_x}, {'targets':test_y}), snapshot_step=500,
#          show_metric=True, run_id=ip.MODEL_NAME)




#Performs Cross Validation
with tf.Session() as session:
  result = cval.cross_validate(session, model, x, y)
  print ('Cross-validation result: %s' % result)
  #print ('Test accuracy: %f' % session.run(accuracy, feed_dict={x: test_x, y: test_y}))

#Saves the model in a different variable
model2 = model

#Saves the model to file
model.save(ip.MODEL_NAME)



#tensorboard --logdir=foo:"D:\Shared_Projects2\Assig 4\Dogs_vs_Cats\Dogs_vs_Cats\log"
