# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:07:41 2017

@author: pc
"""
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.optimizers import rmsprop
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import concatenate
from keras.applications.vgg16 import preprocess_input	
#Data Aug for multi-input
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as Kb
import gc

def VggModel(X_train, layer_number=1, layer_length=19, drop=0.3, opt='Adam'):
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = VGG16(weights='imagenet', include_top=False, 
                 input_shape=X_train.shape[1:], classes=1)
    x = base_model.get_layer('block5_pool').output
    x = GlobalMaxPooling2D()(x)
    merge_one = concatenate([x, angle_layer])
    for layer in base_model.layers[: layer_length]:
        layer.trainable = False
    for i in range(layer_number):
        merge_one = Dense(512, activation='relu')(merge_one)
        merge_one = Dropout(drop)(merge_one)
    predictions = Dense(1, activation='sigmoid')(merge_one)    
    model = Model(input=[base_model.input, input_2], output=predictions)
    if opt == 'Adam':
        optimizer = Adam(lr=1e-4)
    else:
        optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def ResNetModel(X_train, layer_number=1, layer_length=176, drop=0.3, opt='Adam', trainable=False):
    input_2 = Input(shape=[1], name='angle')
    angle_layer = Dense(1, )(input_2)
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=X_train.shape[1:], 
                          classes=1)
    x = base_model.get_layer('avg_pool').output
    x = GlobalMaxPooling2D()(x)
    merge_one = concatenate([x, angle_layer])
    for layer in base_model.layers[: layer_length]:
        layer.trainable = False
    for i in range(layer_number):
        merge_one = Dense(512, activation='relu')(merge_one)
        merge_one = Dropout(drop)(merge_one)
    predictions = Dense(1, activation='sigmoid')(merge_one)
    
    model = Model(input=[base_model.input, input_2], output=predictions)
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-7)
    #opt = Adam(lr = 1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def NormalCNN(X_train):
    inputs = Input(shape=X_train.shape[1:], name="train")
    #model = Sequential()
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)

    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    #x = Dropout(0.2)(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation='relu' )(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu' )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    #x = Dropout(0.2)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    #x = Dropout(0.3)(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    #x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    merge_input = concatenate([x, angle_layer])
    merge = Dense(512, activation='relu')(merge_input)
    merge = Dropout(0.2)(merge)
    merge = Dense(256, activation='relu')(merge)
    merge = Dropout(0.2)(merge)


    predictions = Dense(1, activation='sigmoid')(merge)
    model = Model(input=[inputs, input_2], output=predictions)

    mypotim=Adam(lr=0.01, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer = mypotim, metrics=['accuracy'])
    return model
    