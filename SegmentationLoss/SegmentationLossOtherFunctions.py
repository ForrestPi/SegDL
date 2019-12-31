#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:27:06 2019

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

import pandas as pd
import tensorflow as tf

debug=False

def get_loss_of_segmentation (y_true, y_pred):
    ''' Our goal was to tell different steps, such as dissection, suture, or even more detailed steps, such as dissect bile duct. To approach this, I will first separate one procedure into a series of continuous 5-second segments (or totally 200 segments) with start and end time. The next segment's start time is the previous one's end time. Then, for each segment, I will have a list of binary indicators of 'is_grasper_installed', 'is_grasper_following', 'counts_grasper_following', ('seconds_grasper_following',), 'is_grasper_offscreen', 'counts_grasper_offscreen', ('seconds_grasper_offscreen',) 'is_grasper_activated' (energized), 'counts_grasper_actived', ('seconds_grasper_actived',) (repeat the same thing by replacing grasper->monopolor, bipolar, scissors, clip_applier, needle_drive, suction, energy). For each segment, I will assign a true label from hand labeling. The next step is to implement the loss function, which is a difficult that costs me some time to think. I prefer not the use absolute precision as the loss function, because this way does not punish much on switch to a new label and will cost a lot of noises in the middle of a big chunk of activity. So I would like to implement my loss function this way: after I got the predicted labels, I will take out every time stamp that the label changes between sequential segments. If the change exists in both ground truth and predict, the loss would  add the square of time between the two changes divided by the whole procedure length. Else, for each unexpected changes of labels between sequential segments, the loss function add by 1. In this case, I would punish a lot on these unexpected changes of labels. I will train the model with bi-directional LSTM and RNN, since I believe knowing what happens next matters for the prediction of the previous. At the end, I will group the continuous segments with the same label and return the start and end time of grouped data. He liked my idea. Ben asked me whether I think we have enough data. We now only have 8 Cholecystectomy cases. I told him I would like to try my best. 
    Input:
        y_true: true label, 1D array
        y_pred: predicted label: 1D array
    Output:
        loss: a float value larger than or equal to 0
    
    '''
    #return np.abs(y_true-y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # check the shape of y_true and y_pred.
    if y_true.shape!= y_pred.shape:
        print ('y_true and y_pred are in difference size. ')
        return 0
    
    if len(y_true.shape)!=1:
        print ('y_true and y_pred are not 1d array. ')
        return 0
    
    # hanble when the length of y_true is 0 or 1. 
    if y_true.size==0:
        return 0
    if y_true.size==1:
        if y_true[0]==y_pred[0]:
            return 0
        else:
            return 1
        
    # hanble when the length of y_true is larger than 1. 
    total_length = float(y_true.size-1) # length of the procedure.
    
    # a) find all change values between continuous elements in y_true
    # x is in list_true if the [x]th element and [x+1]th element are different
    # the first element is [0]
    index_true = np.where(y_true[:-1]!=y_true[1:])[0]
    list_true =[]
    for ind in index_true:
        list_true.append({'y_pre': y_true[ind], \
                         'y_post': y_true[ind+1], \
                         'ind': ind, })
    df_true = pd.DataFrame(list_true)
    df_true.sort_values(by=['y_pre', 'y_post', 'ind'], inplace=True)
    df_true.reset_index(drop=True, inplace=True)
    if debug:
        print (df_true)
    # repeat a) for y_pred, and put results in list_pred and df_pred
    index_pred = np.where(y_pred[:-1]!=y_pred[1:])[0]
    list_pred =[]
    for ind in index_pred:
        list_pred.append({'y_pre': y_pred[ind], \
                         'y_post': y_pred[ind+1], \
                         'ind': ind, })
    df_pred = pd.DataFrame(list_pred)
    df_pred.sort_values(by=['y_pre', 'y_post', 'ind'], inplace=True)
    df_pred.reset_index(drop=True, inplace=True)
    if debug:
        print (df_pred)
    
    # compute loss
    loss = 0
    while len(df_true)>0:
        # b) select all rows in df_true that has the same value of y_pre and y_post pair
        y_pre = df_true['y_pre'][0]
        y_post = df_true['y_post'][0]
        if debug:
            print(y_pre, y_post)
        sel_true = (df_true['y_pre']== y_pre) &\
                    (df_true['y_post']== y_post)
        sel_df_true = df_true.loc[sel_true]
        # repeat b) for df_pred
        sel_pred = (df_pred['y_pre']== y_pre) &\
                    (df_pred['y_post']== y_post)
        sel_df_pred = df_pred.loc[sel_pred,:]

        # for the rest of rows in y_true, each row contribute to loss by 1^2
        # these values of y_pre and y_post pair entries
        # exist in y_true not in y_pred
        if len(sel_df_pred)==0:
            loss+=len(sel_df_true)*1**2
        elif len(sel_df_true)==0:
            loss+=len(sel_df_pred)*1**2
            
        else:
            # if the length of the sel_true and sel_pred are the same  
            # return the sum of ( (ind_true-ind_pred)/total_length )**2
            if len(sel_df_true)== len(sel_df_pred):
                loss += np.sum((sel_df_true['ind'].reset_index(drop=True) \
                                       - sel_df_pred['ind'].reset_index(drop=True))**2)/total_length**2
#                print('value:',np.sum((sel_df_true['ind'].reset_index(drop=True) \
#                                       - sel_df_pred['ind'].reset_index(drop=True))**2)/total_length**2)
                if debug: print ('loss0', loss)
            # if the length of the sel_true and sel_pred are not the same
            # use greedy method to decide the loss
            # if the length of the sel_true and sel_pred are not the same
            else: 
                loss += greedy_distance(sel_df_true['ind'].reset_index(drop=True)\
                                        , sel_df_pred['ind'].reset_index(drop=True))/total_length**2\
                        + np.abs(len(sel_df_true)-len(sel_df_pred))*1**2            
                if debug: print ('loss1', loss)
           
        # delete the processed columns in df_true and df_pred
        df_true = df_true.loc[~sel_true, :]
        df_true.reset_index(drop=True, inplace=True)
        df_pred = df_pred.loc[~sel_pred, :]
        df_pred.reset_index(drop=True, inplace=True)        
        
        # for the rest of rows in y_pred, each row contribute to loss by 1^2
        # these values of y_pre and y_post pair entries
        # exist in y_pred not in y_true
        
        if len(df_true) == 0:
            loss+= len(df_pred)*1**2
            if debug: print ('loss2', loss)
            break
    return loss


def greedy_distance(ind_true, ind_pred):
    '''
    The is a way to find the closest match pairs between
    It is done by the following steps:
    1) a matrix A of size (I,J) is created. 
        I is the length of ind_true, J is the length of ind_pred
        element A(ii,jj) = (ind-true[ii]-ind_pred[jj])**2
    2) pick the minimum value of A, add the value to greedy_distance, delete the row and column the value exist 
    
    Input: ind_true, ind_pred
    Output: greedy_distance, float value larger than or equal to 0.
    '''
    greedy_distance = 0
    if debug: print('greedy distance',ind_true, ind_pred)
    if len(ind_true)==0: return 0;
    if len(ind_pred)==0: return 0
    A = np.zeros((len(ind_true), len(ind_pred)))
    for ii in range(len(ind_true)):
        A[ii] = (ind_true[ii]-ind_pred)**2
    if debug: print('A:',A)
    while True:
        # find value and index of the minimum element
        min_element = np.amin(A)
        if debug: print ('min_e:',min_element)
        ii,jj = np.where(A==min_element)[0][0], np.where(A==min_element)[1][0]
        if debug: print('ii,jj:',ii,jj)
        # add the value to greedy_distance
        greedy_distance += min_element
        # delete the row and column the value exist
        try:
            A = np.delete(A,ii, axis=0)
            A = np.delete(A,jj, axis=1)
        except:
            print ('cannot drop row and column')
            break
        if len(A)==0:
            break
        if A.size==0:
            break
    return greedy_distance


