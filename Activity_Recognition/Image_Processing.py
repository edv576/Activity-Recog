import cv2
import numpy as np
import os
import glob
import random
from random import shuffle
from tqdm import tqdm

#Training images directory
TRAIN_DIR = '../data/train'
#Testing images directory
TEST_DIR = '../data/test01'

#Training videos directory
TRAIN_VIDEO_DIR = '../data/ucf-101'

#Tes videos directory
TEST_VIDEO_DIR = '../data/own'

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
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
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
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data

#Gets middle frames from videos and gives them names according to format [Name of activity].[Number of video] _[Number of frame] .jpg
def get_video_frames(route, destRoute):
    f = 0
    for filename in glob.glob(os.path.join(route, '*.avi')):
        vidcap = cv2.VideoCapture(filename)        
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = int(length/2)
        r1 = range(0, middle_frame)
        r2 = range(middle_frame + 1, length - 1)
        second_frame = random.choice(r1)
        third_frame = random.choice(r2)
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        time_length = length/fps
        frame_no = (middle_frame /(time_length*fps))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret,image = vidcap.read()
        my_video_name = filename.split("\\")[-1]
        my_video_name = my_video_name.split(".")[0]
        my_video_name = my_video_name.split("_")[-3]
        my_video_name = my_video_name.lower()
        print(length, middle_frame, second_frame, third_frame, fps, time_length, my_video_name, ret, middle_frame)
        cv2.imwrite(destRoute + '/' + my_video_name + '.' + str(f + 1) + '_' + str(middle_frame) + '.jpg', image)
        f = f + 1
        vidcap.release()

def get_video_frames_test(route, destRoute):
    f = 0
    for filename in glob.glob(os.path.join(route, '*.avi')):
        vidcap = cv2.VideoCapture(filename)        
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = int(length/2)
        r1 = range(0, middle_frame)
        r2 = range(middle_frame + 1, length - 1)
        second_frame = random.choice(r1)
        third_frame = random.choice(r2)

        rval, frame = vidcap.read()
        c = 1
        while rval and c < middle_frame:
            rval, frame = vidcap.read()
            c = c + 1

        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        time_length = length/fps
        #frame_no = (50 /(time_length*fps))
        #vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        #ret,image = vidcap.read()
        my_video_name = filename.split("\\")[-1]
        my_video_name = my_video_name.split(".")[0]
        #my_video_name = my_video_name.split("_")[-3]
        my_video_name = my_video_name.lower()
        print(length, middle_frame, second_frame, third_frame, fps, time_length, my_video_name, rval)
        cv2.imwrite(destRoute + '/' + my_video_name + '.' + str(f + 1) + '_' + str(middle_frame) + '.jpg', frame)
        f = f + 1
        vidcap.release()



#Call the functions to create training and testing data
#training_data = create_train_data()
#testing_data = process_test_data()

#Call the functions to load training and testing data from the previously created files
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

#get_video_frames(TRAIN_VIDEO_DIR, TEST_DIR)
#get_video_frames_test(TEST_VIDEO_DIR, TEST_DIR)







 


