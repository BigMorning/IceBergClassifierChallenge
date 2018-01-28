# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:07:07 2018

@author: pc
"""

import numpy as np
import pandas as pd
import cv2 

def img_reshape(X, w, h):
    X_resize = []
    for i in range(X.shape[0]):
        x = cv2.resize(X[i,:,:], (w, h)).astype(np.float32)
        #x = np.expand_dims(x, axis=0)
        X_resize.append(x)
    X_resize = np.array(X_resize)
    return X_resize

def MaxMinNormalization(x):
    Min = np.min(x)
    Max = np.max(x)
    Mean = np.mean(x)
    x = (x - Mean) / (Max - Min);  
    return x;


def DataProcessingVGG1(train, test):
        #Generate the training data
    X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    #X_band_3=(X_band_1+X_band_2)/2
    X_band_3=np.fabs(np.subtract(X_band_1,X_band_2))
    X_band_4=np.maximum(X_band_1,X_band_2)
    X_band_5=np.minimum(X_band_1,X_band_2)
    
    X_band_3 = MaxMinNormalization(X_band_3)
    X_band_4 = MaxMinNormalization(X_band_4)
    X_band_5 = MaxMinNormalization(X_band_5)
    
    #X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
    X_train = np.concatenate([X_band_3[:, :, :, np.newaxis],X_band_4[:, :, :, np.newaxis],
                              X_band_5[:, :, :, np.newaxis]], axis=-1)
    
    X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    #X_band_test_3=(X_band_test_1+X_band_test_2)/2
    X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
    X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
    X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)
    
    X_band_test_3 = MaxMinNormalization(X_band_test_3)
    X_band_test_4 = MaxMinNormalization(X_band_test_4)
    X_band_test_5 = MaxMinNormalization(X_band_test_5)
    #X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
    X_test = np.concatenate([X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis],
                             X_band_test_5[:, :, :, np.newaxis]],axis=-1)
    return X_train, X_test

def DataProcessingVGG2(train, test, process = True):
    #Generate the training data
    X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_band_3=(X_band_1+X_band_2)/2
    
    if process:
        X_band_1 = MaxMinNormalization(X_band_1)
        X_band_2 = MaxMinNormalization(X_band_2)
        X_band_3 = MaxMinNormalization(X_band_3)
    
    #X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
    X_train = np.concatenate([X_band_1[:, :, :, np.newaxis]
                              , X_band_2[:, :, :, np.newaxis]
                             , X_band_3[:, :, :, np.newaxis]], axis=-1)
    
    X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_band_test_3=(X_band_test_1+X_band_test_2)/2
    #X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
    if process:
        X_band_test_1 = MaxMinNormalization(X_band_test_1)
        X_band_test_2 = MaxMinNormalization(X_band_test_2)
        X_band_test_3 = MaxMinNormalization(X_band_test_3)
     
    X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                              , X_band_test_2[:, :, :, np.newaxis]
                             , X_band_test_3[:, :, :, np.newaxis]], axis=-1)
    return X_train, X_test



def DataProcessingRes1(train, test, process = True):
    w, h = 197, 197
    X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_1 = img_reshape(X_band_1, w, h)
    X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_band_2 = img_reshape(X_band_2, w, h)
    X_band_3=(X_band_1+X_band_2)/2
    
    if process:
        X_band_1 = MaxMinNormalization(X_band_1)
        X_band_2 = MaxMinNormalization(X_band_2)
        X_band_3 = MaxMinNormalization(X_band_3)
    X_train = np.concatenate([X_band_1[:, :, :, np.newaxis]
                          , X_band_2[:, :, :, np.newaxis]
                         , X_band_3[:, :, :, np.newaxis]], axis=-1)
    
    X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_test_1 = img_reshape(X_band_test_1, w, h)
    X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_band_test_2 = img_reshape(X_band_test_2, w, h)
    
    X_band_test_3=(X_band_test_1+X_band_test_2)/2
    
    if process:
        X_band_test_1 = MaxMinNormalization(X_band_test_1)
        X_band_test_2 = MaxMinNormalization(X_band_test_2)
        X_band_test_3 = MaxMinNormalization(X_band_test_3)
     
    X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                              , X_band_test_2[:, :, :, np.newaxis]
                             , X_band_test_3[:, :, :, np.newaxis]], axis=-1)

    return X_train, X_test

def DataProcessingRes2(train, test, process=True):
    w, h = 197, 197
    X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_1 = img_reshape(X_band_1, w, h)
    X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_band_2 = img_reshape(X_band_2, w, h)
    X_band_3=np.fabs(np.subtract(X_band_1,X_band_2))
    X_band_4=np.maximum(X_band_1,X_band_2)
    X_band_5=np.minimum(X_band_1,X_band_2)
    
    if process:
        X_band_3 = MaxMinNormalization(X_band_3)
        X_band_4 = MaxMinNormalization(X_band_4)
        X_band_5 = MaxMinNormalization(X_band_5)
    
    X_train = np.concatenate([X_band_3[:, :, :, np.newaxis], X_band_4[:, :, :, np.newaxis], 
                              X_band_5[:, :, :, np.newaxis]], axis=-1)
    
    X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_test_1 = img_reshape(X_band_test_1, w, h)
    X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_band_test_2 = img_reshape(X_band_test_2, w, h)
    X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
    X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
    X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)
    
    if process:    
        X_band_test_3 = MaxMinNormalization(X_band_test_3)
        X_band_test_4 = MaxMinNormalization(X_band_test_4)
        X_band_test_5 = MaxMinNormalization(X_band_test_5)
    
    X_test = np.concatenate([X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis], 
                             X_band_test_5[:, :, :, np.newaxis]],axis=-1)
    return X_train, X_test