# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:06:34 2017

@author: pc
"""

from util import *
from DataProcess import *
from AllModel import getVGGModelTwo, getResModel
from keras.preprocessing.image import ImageDataGenerator
import gc
import numpy as np
from keras.backend.tensorflow_backend import set_session  
from sklearn.model_selection import StratifiedKFold
import keras.backend as Kb
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import log_loss
import numpy as np
from Model import *


## =========================  Parameters ========================= ##
#文件目录
TRAIN_PATH = "../Data/train.json"
TEST_PATH = "../Data/test.json"
SubName = "../submit/VGG_submit"



#模型选择和数据预处理方式选择
Model_used = VggModel
DataProcess_used = DataProcessingVGG1

batch_size = 64
fold = 4
random_state = 16

#模型可训练的层数
VGG_Train_Layer = [19, 0, 0]
Res_Train_Layer = [176, 0, 0]

train_layer = VGG_Train_Layer
#learning_rate 和 decay

learning_rate = [1e-4, 1e-5, 2e-6]
decay = [1e-7, 1e-8, 3e-9]

#ImageDataGenerator参数
val_horizontal_flip = True
val_vertical_flip = True
val_width_shift_range = 0
val_height_shift_range = 0
val_channel_shift_range = 0
val_zoom_range = 0.5
val_rotation_range = 10
## ==================================================================== ##


## ===========================  ReadData  ============================= ##
train, test = ReadData(TRAIN_PATH, TEST_PATH)
test_id = test['id']
y_train = getTrainLabel(train)
train_anlge, test_angle = GetAngle(train, test)
X_train, X_test = DataProcess_used(train, test)
test_id = test['id']


## ==================================================================== ##


## ===========================  Model Params ============================= ##

gen = ImageDataGenerator(horizontal_flip = val_horizontal_flip,
                         vertical_flip = val_vertical_flip,
                         width_shift_range = val_height_shift_range,
                         height_shift_range = val_channel_shift_range,
                         channel_shift_range=val_channel_shift_range,
                         zoom_range = val_zoom_range,
                         rotation_range = val_rotation_range)

def get_callbacks(filepath, min_lr, patience=10):
   es = EarlyStopping('val_loss', patience=patience, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20,
                          verbose=1,  min_lr = min_lr, mode='min')
   return [es, msave, rl]

def get_callbackTwo(filepath, patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return[es, msave]

def gen_flow_for_two_inputs(gen, batch_size, X1, X2, y):
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=55)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]

## ==================================================================== ##


## =========================  K-FOLD TRAINING ========================= ##

folds = list(StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state).split(X_train, y_train))
iteration = 3
learning_rate = learning_rate
#trainable = [False, True, True]
decay = decay
#min_lr = [1e-5, 1e-5, 1e-5, 1e-7]
y_test_pred_log = 0
y_train_pred_log=0
y_valid_pred_log = 0.0*y_train
train_loss = []
test_loss = []
for j, (train_idx, test_idx) in enumerate(folds):
    print('\n===================FOLD=',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    
    X_holdout = X_train[test_idx]
    Y_holdout= y_train[test_idx]
    
    X_angle_cv = train_anlge[train_idx]
    X_angle_hold=train_anlge[test_idx]
    test_score = []
    for i in range(iteration):
        #tr = trainable[i]
        tr = train_layer[i]
        lr = learning_rate[i]
        dc = decay[i]
        #mr = min_lr[i]
        galaxyModel= Model_used(X_train, layer_number=2, layer_length =tr) 
        Kb.set_value(galaxyModel.optimizer.lr, lr)
        Kb.set_value(galaxyModel.optimizer.decay, dc)
        if i > 0:
            galaxyModel.load_weights(file_path)
        file_path = "%s_%s_aug_model_weights.hdf5"%(j, i)
        #callbacks = get_callbacks(file_path, mr, patience=50)
        callbacks = get_callbackTwo(file_path, patience=20)
        gen_flow = gen_flow_for_two_inputs(gen, batch_size, X_train_cv, X_angle_cv, y_train_cv)
        #Kb.set_value(galaxyModel.optimizer.decay, dc)
        galaxyModel.fit_generator(
                gen_flow,
                steps_per_epoch=24,
                epochs=150,
                shuffle=True,
                verbose=1,
                validation_data=([X_holdout,X_angle_hold], Y_holdout),
                callbacks=callbacks)
        galaxyModel.load_weights(filepath=file_path)
        
        score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        test_score.append(score[0])
        
    best_index = test_score.index(min(test_score))
    file_path = "%s_%s_aug_model_weights.hdf5"%(j, best_index)
    galaxyModel.load_weights(filepath=file_path)
    score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
    print('Best Train loss:', score[0])
    print('Best Train accuracy:', score[1])
    train_loss.append(score[0])
    #Getting Test Score
    score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
    print('Best Test loss:', score[0])
    print('Best Test accuracy:', score[1])
    test_loss.append(score[0])
    
    #Getting validation Score.
    pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])
    y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

    #Getting Test Scores
    temp_test=galaxyModel.predict([X_test, test_angle])
    y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

    #Getting Train Scores
    temp_train=galaxyModel.predict([X_train, train_anlge])
    y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

y_test_pred_log=y_test_pred_log/fold
y_train_pred_log=y_train_pred_log/fold
with open('result.txt', 'a') as f:
    f.write('\n Train Log Loss Validation= ' + str(log_loss(y_train, y_train_pred_log)))
    f.write('\n Test Log Loss Validation= ' + str(log_loss(y_train, y_valid_pred_log)))
    
    f.write('\n Train Log Loss = ' + str(np.mean(train_loss)))
    f.write('\n Test Log Loss = ' + str(np.mean(test_loss)))
    
    f.write('\n each fold train loss ' + str(train_loss))
    f.write('\n each fole test loss ' + str(test_loss))
    
tocsv(y_test_pred_log, test_id, SubName)