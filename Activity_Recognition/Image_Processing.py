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

#Saves dataset for full test videos in 3 parts: 1 for the beginning, 1 for the middle and 1 for the end of every video.
def process_test_data_parts(nParts):
    i = 0
    while i < nParts:
        testing_data = []
        for category in LABELS:
            for img in tqdm(os.listdir(TEST_DIR + category + '/' + str(i + 1))):
                label = label_img(img, category)
                path = os.path.join(TEST_DIR + category + '/' + str(i + 1), img)
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMAGE_SIZE, IMAGE_SIZE))
                testing_data.append([np.array(img), np.array(label)])

        np.save('test_data' + '_' + str(i + 1) + '.npy', testing_data)
        i = i + 1

#Saves dataset for individual test videos
def process_test_video(frames_video, video_name, category):
    testing_data = []
    testing_data_ft = []
    testing_data_st = []
    testing_data_tt = []

    number_frames = len(frames_video)
    first_limit = int(number_frames / 3)
    second_limit = 2 * first_limit

    c = 1
    for frame in frames_video:
        label = label_img(frame, category)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(frame), np.array(label)])
        if(c >= 1 and c <= first_limit):
            testing_data_ft.append([np.array(frame), np.array(label)])
        elif(c > first_limit and c <= second_limit):
            testing_data_st.append([np.array(frame), np.array(label)])
        elif(c > second_limit and c <= number_frames):
            testing_data_tt.append([np.array(frame), np.array(label)])
        c = c + 1



    np.save(FULL_VIDEO_DIR + 'test_data' + '_' + video_name + '_OF' + '.npy', testing_data)
    #np.save(FULL_VIDEO_DIR + 'test_data' + '_' + video_name + '_' + str(1) + '.npy', testing_data_ft)
    #np.save(FULL_VIDEO_DIR + 'test_data' + '_' + video_name + '_' + str(2) + '.npy', testing_data_st)
    #np.save(FULL_VIDEO_DIR + 'test_data' + '_' + video_name + '_' + str(3) + '.npy', testing_data_tt)

    print(video_name)

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
            #my_video_name = filename.split("\\")[-1]
            #my_video_name = my_video_name.split(".")[0]
            #my_video_name = my_video_name.split("_")[-3]
            #my_video_name = my_video_name.lower()

            i = 0

            for imageFrame in frames:
                print(length, middle_frame, second_frame, third_frame, fps, time_length, category, rval)
                cv2.imwrite(TRAIN_DIR + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(i) + '.jpg', imageFrame)
                i = i + 1
            f = f + 1
            vidcap.release()

#Saves all the frames of a video and saves them in individual datasets. Calls process_test_video to do so.
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

#Saves all the frames of a video and saves them in individual datasets, separating them by frames at the beginning/middle/end of a a video.
#Calls process_test_video to do so.
def get_video_frames_test_v_parts(route, destRoute):
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

            number_frames = length
            first_limit = int(number_frames / 3)
            second_limit = 2 * first_limit

            frames = []
            frames_fp = []
            frames_sp = []
            frames_tp = []

            rval, frame = vidcap.read()
            c = 1
            while rval:
                if(c >= 1 and c <= first_limit):
                    frames_fp.append(frame)      
                elif(c > first_limit and c <= second_limit):
                    frames_sp.append(frame) 
                elif(c > second_limit and c <= number_frames):
                    frames_tp.append(frame) 
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
            
            i = 0
            for imageFrame in frames_fp:
                cv2.imwrite(TEST_DIR + '/' + category + '/1/' + category + '.' + str(f + 1) + '_' + str(i) + '_1' + '.jpg', imageFrame)
                i = i + 1

            i = 0
            for imageFrame in frames_sp:
                cv2.imwrite(TEST_DIR + '/' + category + '/2/' + category + '.' + str(f + 1) + '_' + str(i) + '_2' + '.jpg', imageFrame)
                i = i + 1
            
            i = 0
            for imageFrame in frames_tp:
                cv2.imwrite(TEST_DIR + '/' + category + '/3/' + category + '.' + str(f + 1) + '_' + str(i) + '_3' + '.jpg', imageFrame)
                i = i + 1

            f = f + 1
            vidcap.release()

#Gets the middle frames of a test video and saves them in a destiny route. These frames are used latter to create datasets. 
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

#Creates datasets for optical flows for the videos find in origDir. Gives the option to take the middle optical flow or all of them in a video.
#Options: 1 (only the middle optical flow), 2 (all the optical flows)
#separate is a boolean that indicates if the datasets should be created per video or all in one file.
def create_dense_optflow(origDir, destDir, option, separate):
#    if not os.path.exists(TEMP_FOLDER):
#        os.makedirs(TEMP_FOLDER)
    for category in LABELS:
        f = 0
        for filename in glob.glob(os.path.join(origDir + category, '*.avi')):
            cap = cv2.VideoCapture(filename)
    
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = int(length/2)
    
            #cap.get(7)
            #cap.set(1,middle_frame-2);

            ret, frame1 = cap.read() #middle_frame-1

            ofFrames = []

            c = 1
            limit = middle_frame + 1

            if(option == 2):
                limit = length -1




            while ret and c < limit:
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
                frame1 = frame2
                ofFrames.append(rgb)
                c = c + 1

            middle_of = rgb
            
            i = 0

            if(option == 1):
                cv2.imwrite(destDir + '/' + category + '/' + category + '.' + str(f + 1) + '_MF' + '_OF'  + '.jpg', middle_of)
            else:
                for imageFrame in ofFrames:
                    cv2.imwrite(destDir + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(i) + '_OF'  + '.jpg', imageFrame)
                    print(category, f, i)
                    i = i + 1  
                
                if(separate):
                    video_name = category + '.' + str(f + 1)
                    process_test_video(ofFrames, video_name, category)
                    
                                       

            f = f + 1
            cap.release()

#Creates frames for optical flows and then stores them into destDir
def create_dense_optflow_parts(origDir, destDir):
#    if not os.path.exists(TEMP_FOLDER):
#        os.makedirs(TEMP_FOLDER)
    for category in LABELS:
        f = 0
        for filename in glob.glob(os.path.join(origDir + category, '*.avi')):
            cap = cv2.VideoCapture(filename)
    
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = int(length/2)
    
            #cap.get(7)
            #cap.set(1,middle_frame-2);

            ret, frame1 = cap.read() #middle_frame-1

            number_frames = length
            first_limit = int(number_frames / 3)
            second_limit = 2 * first_limit

            ofFrames = []
            ofFrames_fp = []
            ofFrames_sp = []
            ofFrames_tp = []

            c = 1
 


            while ret and c < length - 1:
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
                frame1 = frame2
                ofFrames.append(rgb)
                if(c >= 1 and c <= first_limit):
                    ofFrames_fp.append(rgb)      
                elif(c > first_limit and c <= second_limit):
                    ofFrames_sp.append(rgb) 
                elif(c > second_limit and c <= number_frames):
                    ofFrames_tp.append(rgb) 

                c = c + 1

            middle_of = rgb
            
            i = 0

     
            for imageFrame in ofFrames:
                cv2.imwrite(destDir + '/' + category + '/' + category + '.' + str(f + 1) + '_' + str(i) + '_OF'  + '.jpg', imageFrame)
                print(category, f, i)
                i = i + 1  
            
            i = 0
            for imageFrame in ofFrames_fp:
                cv2.imwrite(destDir + '/' + category + '/1/' + category + '.' + str(f + 1) + '_' + str(i) + '_1' + '.jpg', imageFrame)
                i = i + 1

            i = 0
            for imageFrame in ofFrames_sp:
                cv2.imwrite(destDir + '/' + category + '/2/' + category + '.' + str(f + 1) + '_' + str(i) + '_2' + '.jpg', imageFrame)
                i = i + 1
            
            i = 0
            for imageFrame in ofFrames_tp:
                cv2.imwrite(destDir + '/' + category + '/3/' + category + '.' + str(f + 1) + '_' + str(i) + '_3' + '.jpg', imageFrame)
                i = i + 1 
                    
                                       

            f = f + 1

            cap.release()
           



#Call functions to create optical flows
#create_dense_optflow(TEST_VIDEO_DIR, TEST_DIR, 2, True)
#create_dense_optflow_parts(TEST_VIDEO_DIR, TEST_DIR)
#Call the functions to create training and testing data
#training_data = create_train_data()
#testing_data = process_test_data()


#Call the functions to load training and testing data from the previously created files
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')

#Calls functions to create frames from videos
#get_video_frames(TRAIN_VIDEO_DIR, TRAIN_DIR, 1)
#get_video_frames_test(TEST_VIDEO_DIR, TEST_DIR)
#get_video_frames_test_v(TEST_VIDEO_DIR, TEST_DIR)
#get_video_frames_test_v_parts(TEST_VIDEO_DIR, TEST_DIR)
#process_test_data_parts(3)







 


