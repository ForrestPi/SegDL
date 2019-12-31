#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:24:43 2019

@author: weiji
"""
import random 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
import numpy as np

input_dims=4
time_steps=10
batch_size=5
# the loss is consist of cross entropy loss and label change error loss
factor_ratio = 1. # the ratio of cross entropy loss to label change error loss

def get_loss_of_segmentation_Keras_batch (Tensor_true, Tensor_pred):
    ''' Our goal was to tell different steps, such as dissection, suture, or even more detailed steps, such as dissect bile duct. To approach this, I will first separate one procedure into a series of continuous 5-second segments (or totally 200 segments) with start and end time. The next segment's start time is the previous one's end time. Then, for each segment, I will have a list of binary indicators of 'is_grasper_installed', 'is_grasper_following', 'counts_grasper_following', ('seconds_grasper_following',), 'is_grasper_offscreen', 'counts_grasper_offscreen', ('seconds_grasper_offscreen',) 'is_grasper_activated' (energized), 'counts_grasper_actived', ('seconds_grasper_actived',) (repeat the same thing by replacing grasper->monopolor, bipolar, scissors, clip_applier, needle_drive, suction, energy). For each segment, I will assign a true label from hand labeling. The next step is to implement the loss function, which is a difficult that costs me some time to think. I prefer not the use absolute precision as the loss function, because this way does not punish much on switch to a new label and will cost a lot of noises in the middle of a big chunk of activity. So I would like to implement my loss function this way: after I got the predicted labels, I will take out every time stamp that the label changes between sequential segments. If the change exists in both ground truth and predict, the loss would  add the square of time between the two changes divided by the whole procedure length. Else, for each unexpected changes of labels between sequential segments, the loss function add by 1. In this case, I would punish a lot on these unexpected changes of labels. I will train the model with bi-directional LSTM and RNN, since I believe knowing what happens next matters for the prediction of the previous. At the end, I will group the continuous segments with the same label and return the start and end time of grouped data. He liked my idea. Ben asked me whether I think we have enough data. We now only have 8 Cholecystectomy cases. I told him I would like to try my best. 
    Input:
        y_true: true label, 1D array
        y_pred: predicted label: 1D array
    Output:
        loss: a float value larger than or equal to 0
    
    '''
    #return np.abs(y_true-y_pred)
    
#    y_true = K.argmax(Tensor_true, axis=2)
#    y_pred = K.argmax(Tensor_pred, axis=2)
#    print (y_true.shape)
    y_true = Tensor_true
#    tf.where(
#        tf.equal(tf.reduce_max(Tensor_true, axis=1, keep_dims=True), Tensor_true), 
#        tf.constant(1, shape=Tensor_true.shape), 
#        tf.constant(0, shape=Tensor_true.shape)
#    )
    y_pred = Tensor_pred

    # check the shape of y_true and y_pred.
#    if y_true.shape!= y_pred.shape:
#        print ('y_true and y_pred are in difference size. ')
#        return 0
    
#    if len(y_true.shape)!=1:
#        print ('y_true and y_pred are not 1d array. ')
#        return 0
    
    batch_size_not_used, time_steps_not_used, input_dims_not_used =(y_true.shape)
    
    # currently only work with batch_size 1:
    # I found it difficult to implement custom loss function of various batch size. 
    if batch_size_not_used!=batch_size:
        return tf.convert_to_tensor(0)
    
    # hanble when the length of y_true is 0 or 1. 
    if time_steps==0:
        return tf.convert_to_tensor(0)
    if time_steps==1:
        if y_true==y_pred:
            return tf.convert_to_tensor(0)
        else:
            return tf.convert_to_tensor(1)
        
    
    # hanble when the length of y_true is larger than 1. 
#    total_length = float(time_steps.value-1) # length of the procedure.
    
    #loss = tf.Variable(0, dtype = tf.float32)
    # ii, jj are column num of Input dims.
    # I want to find all indexes that change from ii=1 ->jj=1
    # arr to record loss for each row.
    #arr = tf.TensorArray(dtype = tf.float32, size=100 )#size> input_dims.value*input_dims.value
    #, dynamic_size=True,clear_after_read=False)

    arr = tf.TensorArray(dtype = tf.float32, size=1000 )#size> batch_size,input_dims.value*input_dims.value
    for bb in range(batch_size):
        for ii in range(input_dims):
            for jj in range(input_dims):
                if ii==jj: 
                    arr = arr.write(bb*input_dims*input_dims+ii*input_dims+jj,0)
                    continue
                #print ('ii,jj',ii,jj)
                column_true = tf.slice(y_true,[bb,0,ii],[1,time_steps,1])            
                column_true_ii = K.squeeze(K.squeeze(column_true, 0),1)
                column_true = tf.slice(y_true,[bb,0,jj],[1,time_steps,1])            
                column_true_jj = K.squeeze(K.squeeze(column_true, 0),1)
                index_true = K.tf.where(\
                                tf.logical_and(\
                                    K.tf.equal(tf.slice(column_true_ii, [0,], [time_steps-1,]),1 ),\
                                    K.tf.equal(tf.slice(column_true_jj, [1,], [time_steps-1,]),1 ) \
                                              ))
                len_index_true =index_true.shape[0]
                count_non_zero_index_true = tf.math.count_nonzero(index_true)  

                #print ( 'ii', ii, 'jj', jj,\
                        #'true ii :', column_true_ii.eval(),\
                       #'true jj :', column_true_jj.eval(),\
                       #'ind: ', index_true.eval(),\
                       #'count non zero ind true', count_non_zero_index_true.eval(),\
                       #'count non zero ind pred', count_non_zero_index_true.eval())  

                column_pred = tf.slice(y_pred,[bb,0,ii],[1,time_steps,1])            
                column_pred_ii = K.squeeze(K.squeeze(column_pred, 0),1)
                column_pred = tf.slice(y_pred,[bb,0,jj],[1,time_steps,1])            
                column_pred_jj = K.squeeze(K.squeeze(column_pred, 0),1)
                index_pred = K.tf.where(\
                                tf.logical_and(\
                                    K.tf.equal(tf.slice(column_pred_ii, [0,], [time_steps-1,]),1 ),\
                                    K.tf.equal(tf.slice(column_pred_jj, [1,], [time_steps-1,]),1 ) \
                                              ))
                len_index_pred =index_pred.shape[0]
                count_non_zero_index_pred = tf.math.count_nonzero(index_pred) 

                diff_count_non_zero_index = tf.math.subtract(count_non_zero_index_true, count_non_zero_index_pred)
                abs_diff_count_non_zero_index = tf.math.abs(diff_count_non_zero_index)

                to_add = tf.dtypes.cast(abs_diff_count_non_zero_index , tf.float32)

                arr = arr.write(bb*input_dims*input_dims+ii*input_dims+jj,to_add)

    loss = arr.stack()
    # add custom loss and cross entropy loss
    #return loss 
    return tf.add(tf.reduce_sum(loss),\
        tf.reduce_mean(tf.keras.losses.categorical_crossentropy(\
                Tensor_true, Tensor_pred) )*batch_size*factor_ratio )

 
    
def get_loss_of_segmentation_Keras (Tensor_true, Tensor_pred):
    ''' Our goal was to tell different steps, such as dissection, suture, or even more detailed steps, such as dissect bile duct. To approach this, I will first separate one procedure into a series of continuous 5-second segments (or totally 200 segments) with start and end time. The next segment's start time is the previous one's end time. Then, for each segment, I will have a list of binary indicators of 'is_grasper_installed', 'is_grasper_following', 'counts_grasper_following', ('seconds_grasper_following',), 'is_grasper_offscreen', 'counts_grasper_offscreen', ('seconds_grasper_offscreen',) 'is_grasper_activated' (energized), 'counts_grasper_actived', ('seconds_grasper_actived',) (repeat the same thing by replacing grasper->monopolor, bipolar, scissors, clip_applier, needle_drive, suction, energy). For each segment, I will assign a true label from hand labeling. The next step is to implement the loss function, which is a difficult that costs me some time to think. I prefer not the use absolute precision as the loss function, because this way does not punish much on switch to a new label and will cost a lot of noises in the middle of a big chunk of activity. So I would like to implement my loss function this way: after I got the predicted labels, I will take out every time stamp that the label changes between sequential segments. If the change exists in both ground truth and predict, the loss would  add the square of time between the two changes divided by the whole procedure length. Else, for each unexpected changes of labels between sequential segments, the loss function add by 1. In this case, I would punish a lot on these unexpected changes of labels. I will train the model with bi-directional LSTM and RNN, since I believe knowing what happens next matters for the prediction of the previous. At the end, I will group the continuous segments with the same label and return the start and end time of grouped data. He liked my idea. Ben asked me whether I think we have enough data. We now only have 8 Cholecystectomy cases. I told him I would like to try my best. 
    Input:
        y_true: true label, 1D array
        y_pred: predicted label: 1D array
    Output:
        loss: a float value larger than or equal to 0
    
    '''
    #return np.abs(y_true-y_pred)
    
#    y_true = K.argmax(Tensor_true, axis=2)
#    y_pred = K.argmax(Tensor_pred, axis=2)
#    print (y_true.shape)
    y_true = Tensor_true
#    tf.where(
#        tf.equal(tf.reduce_max(Tensor_true, axis=1, keep_dims=True), Tensor_true), 
#        tf.constant(1, shape=Tensor_true.shape), 
#        tf.constant(0, shape=Tensor_true.shape)
#    )
    y_pred = Tensor_pred

    # check the shape of y_true and y_pred.
#    if y_true.shape!= y_pred.shape:
#        print ('y_true and y_pred are in difference size. ')
#        return 0
    
#    if len(y_true.shape)!=1:
#        print ('y_true and y_pred are not 1d array. ')
#        return 0
    
    batch_size, time_steps_not_used, input_dims_not_used =(y_true.shape)
    
    # currently only work with batch_size 1:
    # I found it difficult to implement custom loss function of various batch size. 
    if batch_size!=1:
        return tf.convert_to_tensor(0)
    
    # hanble when the length of y_true is 0 or 1. 
    if time_steps==0:
        return tf.convert_to_tensor(0)
    if time_steps==1:
        if y_true==y_pred:
            return tf.convert_to_tensor(0)
        else:
            return tf.convert_to_tensor(1)
        
    
    # hanble when the length of y_true is larger than 1. 
#    total_length = float(time_steps.value-1) # length of the procedure.
    
    #loss = tf.Variable(0, dtype = tf.float32)
    # ii, jj are column num of Input dims.
    # I want to find all indexes that change from ii=1 ->jj=1
    # arr to record loss for each row.
    arr = tf.TensorArray(dtype = tf.float32, size=100 )#size> input_dims.value*input_dims.value
    #, dynamic_size=True,clear_after_read=False)
   
    for ii in range(input_dims):
        for jj in range(input_dims):
            if ii==jj: 
                arr = arr.write(ii*input_dims+jj,0)
                continue
            #print ('ii,jj',ii,jj)
            column_true = tf.slice(y_true,[0,0,ii],[1,time_steps,1])            
            column_true_ii = K.squeeze(K.squeeze(column_true, 0),1)
            column_true = tf.slice(y_true,[0,0,jj],[1,time_steps,1])            
            column_true_jj = K.squeeze(K.squeeze(column_true, 0),1)
            index_true = K.tf.where(\
                            tf.logical_and(\
                                K.tf.equal(tf.slice(column_true_ii, [0,], [time_steps-1,]),1 ),\
                                K.tf.equal(tf.slice(column_true_jj, [1,], [time_steps-1,]),1 ) \
                                          ))
            len_index_true =index_true.shape[0]
            count_non_zero_index_true = tf.math.count_nonzero(index_true)  
           
            #print ( 'ii', ii, 'jj', jj,\
                    #'true ii :', column_true_ii.eval(),\
                   #'true jj :', column_true_jj.eval(),\
                   #'ind: ', index_true.eval(),\
                   #'count non zero ind true', count_non_zero_index_true.eval(),\
                   #'count non zero ind pred', count_non_zero_index_true.eval())  

            column_pred = tf.slice(y_pred,[0,0,ii],[1,time_steps,1])            
            column_pred_ii = K.squeeze(K.squeeze(column_pred, 0),1)
            column_pred = tf.slice(y_pred,[0,0,jj],[1,time_steps,1])            
            column_pred_jj = K.squeeze(K.squeeze(column_pred, 0),1)
            index_pred = K.tf.where(\
                            tf.logical_and(\
                                K.tf.equal(tf.slice(column_pred_ii, [0,], [time_steps-1,]),1 ),\
                                K.tf.equal(tf.slice(column_pred_jj, [1,], [time_steps-1,]),1 ) \
                                          ))
            len_index_pred =index_pred.shape[0]
            count_non_zero_index_pred = tf.math.count_nonzero(index_pred) 
            
            diff_count_non_zero_index = tf.math.subtract(count_non_zero_index_true, count_non_zero_index_pred)
            abs_diff_count_non_zero_index = tf.math.abs(diff_count_non_zero_index)
            
            to_add = tf.dtypes.cast(abs_diff_count_non_zero_index , tf.float32)
            
            
            #tf.assign_add(loss, to_add)
        
            arr = arr.write(ii*input_dims+jj,to_add)
    

    loss = arr.stack()
    # add custom loss and cross entropy loss
    #return loss 
    return tf.add(tf.reduce_sum(loss),\
        tf.reduce_mean(tf.keras.losses.categorical_crossentropy(\
                Tensor_true, Tensor_pred) ) )

 

model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences = True), input_shape=(None,2)))
model.add(TimeDistributed(Dense(4, activation='softmax')))
model.compile(loss=get_loss_of_segmentation_Keras, optimizer='adam', metrics=['acc'])