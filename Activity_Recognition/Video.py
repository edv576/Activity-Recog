import cv2

DIR = '../data/test_pic'

vc = cv2.VideoCapture('1_3.avi')
c=1

if vc.isOpened():
    rval , frame = vc.read()

else:
    rval=False
    print("hello")

while rval and c < 10 :
    rval, frame = vc.read()
    cv2.imwrite(DIR + '/' + str(c) + '.jpg',frame)
    c = c + 1
    cv2.waitKey(1)

vc.release()