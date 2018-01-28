# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:43:49 2017

@author: pc
"""
import pandas as pd
import numpy as np
#import cv2

def ReadData(TRAIN_PATH, TEST_PATH):
    train = pd.read_json(TRAIN_PATH)
    test = pd.read_json(TEST_PATH)
    return train, test

def GetAngle(train, test):
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
    train['inc_angle'] = train['inc_angle'].fillna(method='pad') 
    #'pad'用前一个数填充na
    train_anlge = train['inc_angle']
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    test_angle = test['inc_angle']
    return train_anlge, test_angle

def getTrainLabel(train):
    y_train=train['is_iceberg']
    return y_train

def getImageData(train, test, mode=1): 
    if mode ==1:
        X_train, X_test = DataProcessingMode1(train, test)
    else:
        X_train, X_test = DataProcessingResMode(train, test)
    return X_train, X_test

def tocsv(preds, test_id, subname):
    submission = pd.DataFrame()
    submission['id']=test_id
    submission['is_iceberg']=preds
    submission.to_csv(subname, index=False)



