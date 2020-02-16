import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

smooth = 1.

# def iou(y_true, y_pred):
#     y_pred = tf.cast(y_pred[:,:,:,0],'float32')
#     y_true = tf.cast(y_true[:,:,:,0],'float32')   
    
# #     pred = tf.cast(y_pred,'float32')
# #     true = tf.cast(y_true,'float32')  
     
#     y_pred = keras.batch_flatten(y_pred)
#     y_true = keras.batch_flatten(y_true)
    
#     tp = y_true * y_pred
#     fp = y_true * (1 - y_pred)
#     fn = (1-y_true) * y_pred
    
#     tp = keras.sum(tp,axis=-1)
#     fp = keras.sum(fp,axis=-1)
#     fn = keras.sum(fn,axis=-1)
    
#     intersection = tp + smooth 
#     union= (tp+fp+fn) + smooth
    
#     iou = intersection/union
#     return iou

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)    
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def res_unet(pretrained_weights = None, batchnorm=True, input_size = (256,256,1)):
    
    inputs = Input(input_size)
    
    #Downsample Block 1
    shortcut = Conv2D(64, (3, 3), padding='same')(inputs)
    shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
    
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    if batchnorm == True:
        conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    if batchnorm == True:
        conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    
    conv1 = Add()([shortcut,conv1])
    conv1 = Activation('relu')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    #Downsample Block 2
    shortcut = Conv2D(128, (3, 3), padding='same')(pool1)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
       
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    if batchnorm == True:    
        conv2 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    if batchnorm == True:    
        conv2 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv2)
    
    conv2_pre_act = Add()([shortcut,conv2])
    conv2 = Activation('relu')(conv2_pre_act)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #Downsample Block 3    
    shortcut = Conv2D(256, (3, 3), padding='same')(pool2)
    if batchnorm == True:    
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
       
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    if batchnorm == True:    
        conv3 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
    if batchnorm == True:
        conv3 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv3)
    
    conv3 = Add()([shortcut,conv3])
    conv3 = Activation('relu')(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    #Downsample Block 4
    shortcut = Conv2D(512, (3, 3), padding='same')(pool3)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
       
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    if batchnorm == True:
        conv4 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv4)
    conv4 = Activation('relu')(conv4)
    
    conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
    if batchnorm == True:
        conv4 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv4)
    
    conv4 = Add()([shortcut,conv4])
    conv4 = Activation('relu')(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)    
    
    #Middle block    
    shortcut = Conv2D(1024, (3, 3), padding='same')(pool4)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
       
    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    if batchnorm == True:
        conv5 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv5)
    conv5 = Activation('relu')(conv5)
    
    conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
    if batchnorm == True:
        conv5 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv5)
    
    conv5 = Add()([shortcut,conv5])
    conv5 = Activation('relu')(conv5)  
    
    #Upsample Block 1
    
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)   
    if batchnorm == True:
        up6 = BatchNormalization(axis=3, epsilon=1.001e-5)(up6)  
    up6 = Activation('relu')(up6)
    up6 = concatenate([up6, conv4], axis=3)
        
    shortcut = Conv2D(512, (3, 3), padding='same')(up6)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
    
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    if batchnorm == True:
        conv6 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv6)
    conv6 = Activation('relu')(conv6)
    
    conv6 = Conv2D(512, (3, 3), padding='same')(conv6)
    if batchnorm == True:
        conv6 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv6)
    
    conv6 = Add()([shortcut,conv6])
    conv6 = Activation('relu')(conv6)      

    #Upsample Block 2    
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    if batchnorm == True:
        up7 = BatchNormalization(axis=3, epsilon=1.001e-5)(up7)   
    up7 = Activation('relu')(up7)    
    up7 = concatenate([up7, conv3], axis=3)
        
    shortcut = Conv2D(256, (3, 3), padding='same')(up7)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
    
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    if batchnorm == True:
        conv7 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv7)
    conv7 = Activation('relu')(conv7)
    
    conv7 = Conv2D(256, (3, 3), padding='same')(conv7)
    if batchnorm == True:
        conv7 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv7)
    
    conv7 = Add()([shortcut,conv7])
    conv7 = Activation('relu')(conv7)         
   
    #Upsample Block 3
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    if batchnorm == True:
        up8 = BatchNormalization(axis=3, epsilon=1.001e-5)(up8)  
    up8 = Activation('relu')(up8)
    up8 = concatenate([up8, conv2], axis=3)
        
    shortcut = Conv2D(128, (3, 3), padding='same')(up8)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
    
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    if batchnorm == True:
        conv8 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv8)
    conv8 = Activation('relu')(conv8)
    
    conv8 = Conv2D(128, (3, 3), padding='same')(conv8)
    if batchnorm == True:
        conv8 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv8)
    
    conv8 = Add()([shortcut,conv8])
    conv8 = Activation('relu')(conv8)       

    #Upsample Block 4
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    if batchnorm == True:
        up9 = BatchNormalization(axis=3, epsilon=1.001e-5)(up9)   
    up9 = Activation('relu')(up9)
    up9 = concatenate([up9, conv1], axis=3)
        
    shortcut = Conv2D(64, (3, 3), padding='same')(up9)
    if batchnorm == True:
        shortcut = BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)
    
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    conv9 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv9)
    conv9 = Activation('relu')(conv9)
    
    conv9 = Conv2D(64, (3, 3), padding='same')(conv9)
    if batchnorm == True:
        conv9 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv9)
    
    conv9 = Add()([shortcut,conv9])
    conv9 = Activation('relu')(conv9)         
       
    #Output
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = ['accuracy', iou])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


