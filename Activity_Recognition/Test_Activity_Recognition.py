import Image_Processing as ip
import Model_Activity_Recognition as mar
import matplotlib.pyplot as plt
import numpy as np

#Gets test data. In our case should the data from our videos
test_data = ip.process_test_data()

fig = plt.figure()

#Iterates over the number of examples we want to test
#[:12] means take all examples from the beginning to the number 12
for num, data in enumerate(test_data):
    img_num = data[1]
    img_data = data[0]
    
    
    y = fig.add_subplot(5, 7, num + 1)
    orig = img_data
    data = img_data.reshape(ip.IMAGE_SIZE, ip.IMAGE_SIZE, 1)
    
    #Predicts the label for one of the test images
    model_out =  mar.model2.predict([data])[0]

    #The category is decided
    if(np.argmax(model_out) == 0):
        str_label = 'Brushing teeth'
    elif(np.argmax(model_out) == 1):
        str_label = 'Cutting in Kitchen'
    elif(np.argmax(model_out) == 2):
        str_label = 'Jumping Jack'
    elif(np.argmax(model_out) == 3):
        str_label = 'Lunges'
    elif(np.argmax(model_out) == 4):
        str_label = 'Wall Pushups'

    #Draws the test image in greyscale
    y.imshow(orig, cmap='gray')
    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
      
