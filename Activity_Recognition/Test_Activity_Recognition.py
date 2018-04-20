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
    img_cat = str(data[1])
    img_data = data[0]
    
    
    y = fig.add_subplot(5, 7, num + 1)
    orig = img_data
    data = img_data.reshape(ip.IMAGE_SIZE, ip.IMAGE_SIZE, 3)
    
    #Predicts the label for one of the test images
    model_out =  mar.final_model.predict([data])[0]

    #The category is decided
    if(np.argmax(model_out) == 0):
        str_label = 'Brushing Teeth'
        if(img_cat.lower() == 'brushingteeth'):
            str_label = str_label + 'True'
        else:
            str_label = str_label + 'False'    

    elif(np.argmax(model_out) == 1):
        str_label = 'Cutting in Kitchen'
        if(img_cat.lower() == 'cuttinginkitchen'):
            str_label = str_label + 'True'
        else:
            str_label = str_label + 'False' 

    elif(np.argmax(model_out) == 2):
        str_label = 'Jumping Jack'
        if(img_cat.lower() == 'jumpingjack'):
            str_label = str_label + 'True'
        else:
            str_label = str_label + 'False' 
    elif(np.argmax(model_out) == 3):
        str_label = 'Lunges'
        if(img_cat.lower() == 'lunges'):
            str_label = str_label + 'True'
        else:
            str_label = str_label + 'False' 

    elif(np.argmax(model_out) == 4):
        str_label = 'Wall Pushups'
        if(img_cat.lower() == 'wallpushups'):
            str_label = str_label + 'True'
        else:
            str_label = str_label + 'False' 

    #Draws the test image in greyscale
    y.imshow(orig, cmap='gray')
    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
      
