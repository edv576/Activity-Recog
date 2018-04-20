import cv2
import numpy as np
import os
import glob
import random
from random import shuffle
from tqdm import tqdm

#Training images directory
TRAIN_DIR = '../data/train/'
#Testing images directory
TEST_DIR = '../data/test01/'

#Training videos directory
TRAIN_VIDEO_DIR = '../data/ucf-101/'

#Test videos directory
TEST_VIDEO_DIR = '../data/own/'

TEMP_FOLDER = '../data/temp_optf'

#Size of the image
IMAGE_SIZE = 120
#Learning rate to use
LR = 1e-3

#Given name to model
MODEL_NAME = 'activity_recognition-{}-{}.model'.format(LR, '6conv-basic-video')

#Labels
LABELS = ['brushingteeth', 'cuttinginkitchen', 'jumpingjack', 'lunges', 'wallpushups']

#According to the name of image, get its label
#Since there are five clases an array of five is used. The number one is used to define the index corresponding to the label
def label_img(categ):
    if categ == 'brushingteeth': return[1, 0, 0, 0, 0] 
    elif categ == 'cuttinginkitchen': return[0, 1, 0, 0, 0]
    elif categ == 'jumpingjack': return[0, 0, 1, 0, 0]
    elif categ == 'lunges': return[0, 0, 0, 1, 0]
    elif categ == 'wallpushups': return[0, 0, 0, 0, 1]

#Takes training images and turn them into a .npy file to latter use them in the model
def create_train_data():
    training_data = []
    for category in LABELS:
        for img in tqdm(os.listdir(TRAIN_DIR + category)):
            label = label_img(category)
            path = os.path.join(TRAIN_DIR + category, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#Takes training images and turn them into a .npy file to latter use them in the model
def process_test_data():
    testing_data = []
    for category in LABELS:
        for img in tqdm(os.listdir(TEST_DIR + category)):
            label = label_img(category)
            path = os.path.join(TEST_DIR + category, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
            testing_data.append([np.array(img), np.array(label)])

    np.save('test_data.npy', testing_data)
    return testing_data

#def process_test_data():
#    testing_data = []
#    for category in LABELS:
#        for img in tqdm(os.listdir(TEST_DIR + category)):
#            path = os.path.join(TEST_DIR + category, img)
#            img_cat = category
#            img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
#            testing_data.append([np.array(img), img_cat])

#    np.save('test_data.npy', testing_data)
#    return testing_data

#Gets middle frames from videos and gives them names according to format [Name of activity].[Number of video] _[Number of frame] .jpg
def get_video_frames(route, destRoute):
    for category in LABELS:
        f = 0
        for filename in glob.glob(os.path.join(TRAIN_VIDEO_DIR + category, '*.avi')):
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
            #frame_no = (middle_frame /(time_length*fps))
            #vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            #ret,image = vidcap.read()
            my_video_name = filename.split("\\")[-1]
            my_video_name = my_video_name.split(".")[0]
            my_video_name = my_video_name.split("_")[-3]
            my_video_name = my_video_name.lower()
            print(length, middle_frame, second_frame, third_frame, fps, time_length, category, rval, middle_frame)
            cv2.imwrite(TRAIN_DIR + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(middle_frame) + '.jpg', frame)
            f = f + 1
            vidcap.release()

def get_video_frames_test(route, destRoute):
    for category in LABELS:
        f = 0
        for filename in glob.glob(os.path.join(TEST_VIDEO_DIR + category, '*.avi')):
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
            cv2.imwrite(TEST_DIR + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(middle_frame) + '.jpg', frame)
            f = f + 1
            vidcap.release()



#Call the functions to create training and testing data
#training_data = create_train_data()
#testing_data = process_test_data()

#Call the functions to load training and testing data from the previously created files
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')

#get_video_frames(TRAIN_VIDEO_DIR, TEST_DIR)
#get_video_frames_test(TEST_VIDEO_DIR, TEST_DIR)

def create_dense_optflow(video):
#    if not os.path.exists(TEMP_FOLDER):
#        os.makedirs(TEMP_FOLDER)
        
    cap = cv2.VideoCapture(video)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = int(length/2)
    
    cap.get(7)
    cap.set(1,middle_frame-2);

    ret, frame1 = cap.read() #middle_frame-1
    frame1 = cv2.resize(frame1, (IMAGE_SIZE, IMAGE_SIZE)) 
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (IMAGE_SIZE, IMAGE_SIZE)) 
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    #cv2.imwrite(TEMP_FOLDER + '/' + str(i) + '.jpg',rgb)
    cap.release()
    return rgb
    

def create_optflows_data(folder_data):
    opticalflows = []
    i = 1
    for category in LABELS:
        for video in glob.glob(os.path.join(folder_data + category, '*.avi')):
            print(i, '. ', video)
            img = create_dense_optflow(video)
            label = label_img(category)
            opticalflows.append([np.array(img), np.array(label)])
            i = i + 1
    
    np.save('train_optical_data.npy', opticalflows)
    print('Created train_optical_data.npy')
    return opticalflows


create_optflows_data(TRAIN_VIDEO_DIR)

 


