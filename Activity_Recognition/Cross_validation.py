import numpy as np
import Image_Processing as ip
import tflearn as tfl
from sklearn.model_selection import KFold # import KFold
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#Performs Cross validation for the model
#Inputs: Model(model data), train_x_all(input data), train_y_all(labels data) and split_size(number of folds)
#The session input can be eliminated since its not used
def cross_validate(session, model, train_x_all, train_y_all, split_size=10):
  results = []
  kf = KFold(n_splits=split_size)
  c = 1
  #Iterates over the fold configuration
  for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    print('Validation:', c)
    train_x = train_x_all[train_idx]
    #print(train_x)
    train_y = train_y_all[train_idx]
    val_x = train_x_all[val_idx]
    val_y = train_y_all[val_idx]
    
    #Trains model using one of the fold configuration
    #model.fit({'input':train_x}, {'target':train_y}, n_epoch=5, validation_set=({'input':val_x}, {'target':val_y}), snapshot_step=500,
    #      show_metric=True, run_id=ip.MODEL_NAME)
    model.fit(train_x, train_y, n_epoch=5, validation_set=({'input':val_x}, {'target':val_y}), snapshot_step=500,
          show_metric=True, run_id=ip.MODEL_NAME)

    #print(tfl.Accuracy())

    #Add results of accuracy to array
    results.append(model.evaluate(val_x, val_y))
    c = c + 1

  return results