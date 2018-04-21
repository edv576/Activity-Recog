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

#Full video dataset directory
FULL_VIDEO_DIR = '../data/full_video/'

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
def label_img(img, categ):
    #word_label = img.split('.')[-3]
    #word_label = word_label.lower()
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
            label = label_img(img, category)
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
            label = label_img(img, category)
            path = os.path.join(TEST_DIR + category, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
            testing_data.append([np.array(img), np.array(label)])

    np.save('test_data.npy', testing_data)
    return testing_data

def process_test_video(frames_video, video_name, category):
    testing_data = []

    for frame in frames_video:
        label = label_img(frame, category)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(frame), np.array(label)])

    np.save(FULL_VIDEO_DIR + 'test_data' + '_' + video_name + '.npy', testing_data)
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
#inputs: type (type of frame extraction 1-One 2-Multiple 3-One and mirrored 4-Multiple and mirrored) 
def get_video_frames(route, destRoute, type):
    for category in LABELS:
        f = 0
        for filename in glob.glob(os.path.join(TRAIN_VIDEO_DIR + category, '*.avi')):
            vidcap = cv2.VideoCapture(filename)        
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = int(length/2)
            r1 = range(3, middle_frame)
            print(r1)
            r2 = range(middle_frame + 1, length - 1)    
            print(r2)      
            second_frame = random.choice(r1)
            third_frame = random.choice(r2)
            r3 = range(2, second_frame)
            print(r3)
            r4 = range(third_frame + 1, length)
            print(r4)
            fourth_frame = random.choice(r3)
            fifth_frame = random.choice(r4)

            nframes = [middle_frame, second_frame, third_frame, fourth_frame, fifth_frame]
            frames = []

            rval, frame = vidcap.read()
            c = 1

            if(type == 1 or type == 3):
                while rval and c < middle_frame + 1:
                    rval, frame = vidcap.read()
                    if(c == middle_frame):
                        frames.append(frame)
                        if(type == 3):
                            fframe = cv2.flip(frame, 1)
                            frames.append(fframe)
                    c = c + 1

            if(type == 2 or type == 4):
                while rval:
                    rval, frame = vidcap.read()
                    if(nframes.count(c) > 0):
                        frames.append(frame)
                        if(type == 4):
                            fframe = cv2.flip(frame, 1)
                            frames.append(fframe)
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

            i = 0

            for imageFrame in frames:
                print(length, middle_frame, second_frame, third_frame, fps, time_length, category, rval)
                cv2.imwrite(TRAIN_DIR + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(i) + '.jpg', imageFrame)
                i = i + 1
            f = f + 1
            vidcap.release()

def get_video_frames_test_v(route, destRoute):
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

            frames = []

            rval, frame = vidcap.read()
            c = 1
            while rval:
                frames.append(frame)
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
            video_name = category + '.' + str(f + 1)
            print(length, fps, time_length, video_name, rval)
            process_test_video(frames, video_name, category)

            i = 0

            for imageFrame in frames:
                cv2.imwrite(TEST_DIR + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(i) + '.jpg', imageFrame)
                i = i + 1

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
testing_data = process_test_data()

#Call the functions to load training and testing data from the previously created files
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')

#get_video_frames(TRAIN_VIDEO_DIR, TRAIN_DIR, 4)
#get_video_frames_test(TEST_VIDEO_DIR, TEST_DIR)
#get_video_frames_test_v(TEST_VIDEO_DIR, TEST_DIR)







 


