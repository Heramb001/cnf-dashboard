# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:36:48 2020

@author: hpendyal
"""

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
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

# Load .h5 keras/tensorflow model

def load_model(model):

    model = tf.keras.models.load_model('mldata/%s.h5'%model) 
    return model

# Get the saved scaler

# Add gaussian noise

def addNoise(data, alpha, num_features):

    predList = []
    for i in range(1000):
        predList.append(data + (alpha * np.random.rand(num_features,)))
    predList = np.array(predList)

    return predList


def backwardPredict(fname, model, ):
# load xsec file
    xsec = np.load('mldata/%s.npy'%fname)
    ml = load_model(model) 


    # add 0.05 noise 
    predList = addNoise(xsec, 0.05, 342) 

    # make the prediction
    pred = ml.predict(predList)
    #pred = par_scaler.inverse_transform(pred)
    np.save('data/%s-par.npy'%fname, pred)
    return pred



if __name__=='__main__':

    fname = 'test_backward'
    modelname = 'model_1'

    pred = backwardPredict(fname, modelname)
    print("\nSaving predicted parameters file in example1/data folder ...")