# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:36:48 2020

@author: hpendyal
"""

import numpy as np
from tensorflow.keras.models import load_model
import mdn
import tensorflow as tf
import pandas as pd
#from sklearn.externals import joblib
# from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import h5py
#from tools import load,save,checkdir
#import theory
import os
import matplotlib.pyplot as plt

from utils.scalerUtils import unscaleOutput

#dataDir = '../'
dataDir = ''
# Load .h5 keras/tensorflow model
def load_model(model):
    output_dim= 10
    num_mixes= 1
    model = tf.keras.models.load_model(dataDir+'models/%s.h5'%model, 
                                       custom_objects=
                                       {
                                               'MDN':mdn.MDN,
                                               'mdn_loss_func':mdn.get_mixture_loss_func(output_dim,num_mixes)
                                        }) 
    return model

# Get the saved scaler

# Add gaussian noise

'''
def addNoise(data, alpha, num_features):

    predList = []
    for i in range(1000):
        predList.append(data + (alpha * np.random.rand(num_features,)))
    predList = np.array(predList)

    return predList
'''

def addNoiseSelected(data, n_value, uncertainity_value, validation_column, data_column, num_features):
    predList = []
    #--- formula : xsec + Randomnumber * uncertainity * noise_value
    for i in range(1000):
        predList.append(np.where(data[validation_column]==True, #--- if data is selected
                                 data[data_column] + (float(uncertainity_value) * np.random.rand(num_features,) * float(n_value)), #---- add value (noise * uncertainity * random number)
                                 data[data_column])) #--- else remain the data as it is
        #predList.append(data + (alpha * np.random.rand(num_features,)))
    predList = np.array(predList)
    return predList

def getCrossection(obs_p, obs_n):
    return np.append(obs_p,obs_n)

def addNoise(data):
    predList = []
    #--- formula 
    #--- for the selected_data : xsec + Randomnumber * uncertainity * noise_value
    for i in range(1000):
        #--- create a pseudo table 
        data_copy = data.copy()
        #--- modify columns
        data_copy['obs_p'] += np.random.randn(len(data_copy['obs_p'])) * data_copy['err_obs_p_mod']
        data_copy['obs_n'] += np.random.randn(len(data_copy['obs_n'])) * data_copy['err_obs_n_mod']
        predList.append(getCrossection(data_copy['obs_p'],data_copy['obs_n']))
    predList = np.array(predList)
    #print('---> (RUNLOG) -- Shape of Crossection :  ',predList.shape)
    return predList
    
def calculate_xsec(p_Data, n_Data):
    return np.concatenate((p_Data, n_Data), axis = 1)

def backwardPredict(fname, model, xsec_noised):
    # load xsec file
    #xsec = calculate_xsec(dataframe)
    ml = load_model(model) 
    # make the prediction
    pred = ml.predict(xsec_noised)
    pred = pred[1]
    #pred = par_scaler.inverse_transform(pred)
    pred = unscaleOutput(pred)
    np.save(dataDir+'data/%s-par.npy'%fname, pred)
    return pred



if __name__=='__main__':

    fname = 'test_backward'
    modelname = 'model_1'
    dataFrame = pd.DataFrame()
    pred = backwardPredict(fname, modelname, 0.0132, dataFrame)
    print("\nSaving predicted parameters file in example1/data folder ...")