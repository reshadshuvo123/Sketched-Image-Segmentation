import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import os
import time
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from PIL import Image,ImageDraw
import time
from skimage.io import imread_collection

os.environ['MKL_THREADING_LAYER'] = 'GNU'

data = '/home/reshad/Medical-Transformer/Images_100/'
result_location = '/home/reshad/Medical-Transformer/test_folder'
result_location_img= '/home/reshad/Medical-Transformer/test_folder/img/'
result_location_mask = '/home/reshad/Medical-Transformer/test_folder/labelcol/'
result_location_orginal = '/home/reshad/Medical-Transformer/test_folder/orginal/'

def preprocessing(data, result_location_img,result_location_mask):
    print('************************* Preprocessing start and Mask Generation using Point-shooting Method ***************************')
    count = 0
    data_list = os.listdir(data)
    print('\n')
    print('Total Number of Images : ', len(data_list))
    print('\n')
    for i in range(len(data_list)):
        print('Start Preprocessing Image No :', i+1) 
        # Load image, grayscale, blur, Otsu's threshold
        os.chdir(data)
        img = cv2.imread(data_list[i])  ### reading image
        image=orginal=img.copy()
        #### Saving original images #####
        os.chdir(result_location_orginal)
        cv2.imwrite("orginal_%d.png" % count, orginal )  ### Saving Masked Image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### gray conversion
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  ## making 0 or 255 pixel
        ### Black pixel co-ordinate detection
        ii1 = np.nonzero(thresh == 255)
        x = list(ii1[1])
        y = list(ii1[0])
        # Now, loop through coord arrays, and create a circle at each x,y pair
        os.chdir(result_location)
        for xx, yy in zip(x, y):
            cv2.circle(img, (xx, yy), 10, (0, 20, 200), 10)
        #### Saving input images #################
        os.chdir(result_location_img)
        orginal= resize(image, (128, 128, 3), mode='constant', preserve_range=True)
        cv2.imwrite("orginal_%d.png" % count, orginal)  ### Saving resized original Image
        img = resize(img, (128, 128), mode='constant', preserve_range=True) ## resize mask
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### gray conversion
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #print(np.unique(mask))
        #print(mask.shape, orginal.shape)
        ### Saving the mask images ##############
        os.chdir(result_location_mask)
        cv2.imwrite("mask_%d.png" % count, mask)  ### Saving Masked Image
        count=count + 1
    print('*********************************** Preprocessing done *************************')


def rename(result_location_img):
    collection = result_location_img
    for i, filename in enumerate(os.listdir(collection)):
        p= str(i).zfill(4)
        os.rename(collection + filename, collection + str(p) + ".png")

preprocessing(data, result_location_img,result_location_mask)
rename(result_location_img)
rename(result_location_mask)
rename(result_location_orginal)

 
print('**************************************** Generation of Transformer based Segmentation Masks Started  ****************************')
print('\n')
start_time = time.time()
os.chdir('/home/reshad/Medical-Transformer')
os.system('python test.py --loaddirec "/home/reshad/Medical-Transformer/train_results/390/MedT.pth" --val_dataset "/home/reshad/Medical-Transformer/test_folder" --direc "/home/reshad/Medical-Transformer/test_results" --batch_size 1 --modelname "MedT" --imgsize 128 --gray "no"')
print('\n')
print('**************************************** Time Required for Transformer to Generate Segmentation Masks ****************************')
print('\n')
print("--- %s seconds ---" % (time.time() - start_time))

orginal_path = '/home/reshad/Medical-Transformer/test_folder/orginal/*.png'
prediction_path='/home/reshad/Medical-Transformer/test_results/*.png'
result_path='/home/reshad/Medical-Transformer/test_results_orginal'
orginal_images=X_valid= imread_collection(orginal_path)
prediction_images= preds_val=imread_collection(prediction_path)
os.chdir(result_path)


def resize_boundingbox(X_valid, preds_val):
    ROI_number = 0
    for i in range(len(X_valid)):
        print('\n')
        print('Image is going to processed : ', i)
        print('\n')
        img = X_valid[i].copy()
        preds=preds_val[i]
        orx=img.shape[1]
        ory=img.shape[0]
        scalex=orx/128
        scaley= ory/128
        # Perform a little bit of morphology:
        # Set kernel (structuring element) size:
        kernelSize = (3, 3)
        # Set operation iterations:
        opIterations = 1
        # Get the structuring element:
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        # Perform Dilate:
        morphology = cv2.morphologyEx(preds, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
        contours, hierarchy = cv2.findContours(morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        im=X_valid[i].copy()
        all_coordinates=[]
        for c in contours:
            rect = cv2.boundingRect(c)
            if rect[2] < 5 or rect[3] < 5: continue
            cv2.contourArea(c)
            x, y, w, h = rect
            x=int(x*scalex)
            y=int(y*scaley)
            w= int(w* scalex)
            h = int(h * scaley)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ROI = im[y:y + h, x:x + w]
            cv2.imwrite('ROI_{}_{}.png'.format(i, ROI_number), ROI)
            ROI_number = ROI_number + 1
            coordinates=(y,y+h,x,x+w)
            all_coordinates.append(coordinates)
        cv2.imwrite("result_bounding_box_%d.png" % i, im)
        ROI_number = 0
        with open('coordinate_file_%d.txt'%i, 'w') as fh:
            for o in range(len(all_coordinates)):
                fh.write('{}_{} \n'.format(o,all_coordinates[0]))

print('**************************** Resize the bounding Box ******************************* ')
resize_boundingbox(X_valid, preds_val)

