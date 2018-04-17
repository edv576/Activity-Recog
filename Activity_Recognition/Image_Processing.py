import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

#Training images directory
TRAIN_DIR = 'D:/Shared_Projects2/Assig 4/Activity_Recognition/Activity_Recognition/train'
#Testing images directory
TEST_DIR = 'D:/Shared_Projects2/Assig 4/Activity_Recognition/Activity_Recognition/test1'

#Size of the image
IMAGE_SIZE = 120
#Learning rate to use
LR = 1e-3

#Given name to model
MODEL_NAME = 'activity_recognition-{}-{}.model'.format(LR, '6conv-basic-video')

#According to the name of image, get its label
#Since there are five clases an array of five is used. The number one is used to define the index corresponding to the label
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'brushingteeth': return[1, 0, 0, 0, 0] 
    elif word_label == 'cuttinginkitchen': return[0, 1, 0, 0, 0]
    elif word_label == 'jumpingjack': return[0, 0, 1, 0, 0]
    elif word_label == 'lunges': return[0, 0, 0, 1, 0]
    elif word_label == 'wallpushups': return[0, 0, 0, 0, 1]

#Takes training images and turn them into a .npy file to latter use them in the model
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#Takes training images and turn them into a .npy file to latter use them in the model
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data

#Call the functions to create training and testing data
training_data = create_train_data()
testing_data = process_test_data()

#Call the functions to load training and testing data from the previously created files
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')







 


