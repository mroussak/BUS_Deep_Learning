from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2


def train_generator_npy(input_shape,image_path,mask_path,image_prefix,mask_prefix):    
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        print('loading :,',item)
        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, input_shape)
#         img_input = (img-np.mean(img))/np.std(img)
        img_input = img/255.0
        img_input = np.reshape(img_input,img_input.shape + (1,))
        print('img shape :', img_input.shape)
        or_mask = cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),cv2.IMREAD_GRAYSCALE)
        mask = np.where(or_mask==255.0,1.0,0.0).astype(np.uint8)
        mask = cv2.resize(mask, input_shape).astype(or_mask.dtype)
        mask = np.reshape(mask,mask.shape + (1,))
        print('mask shape :', mask.shape)
        print('mask pixel dist :', np.unique(mask,return_counts=True))
        print('')
        image_arr.append(img_input)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def convert_mask_generator(generator):
    while True:
        xi, yi = next(generator)
        yn = np.concatenate([yi, 1.0-yi], axis=-1)  
        yield(xi, yn)