# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:47:49 2020

@author: hpendyal
"""

import pickle
import numpy as np
import mdn
#--- import config to get all external dependencies
from config import Config
#--- create Config Class
config = Config()

#dataDir = '../'
dataDir = ''
#--- function to scale the data and return obs_p, obs_n
def scaleCross(xsecData, selectedSample):
    crossScaler = pickle.load(open(dataDir+config.xsecScalerPath, 'rb'))
    #--- scale the cross section
    scaledXsec = crossScaler.transform(xsecData)
    obs_p,obs_n = np.split(scaledXsec[selectedSample],2)
    return obs_p,obs_n

def unscaleOutput(predData):
    parScaler = pickle.load(open( dataDir+config.outScalerPath,'rb'))
    nn_pred= np.apply_along_axis(mdn.sample_from_output,1,predData, 10, 1,temp=1.0, sigma_temp=0.05)
    nn_pred= nn_pred[:,0,:]
    nn_pred= parScaler.inverse_transform(nn_pred) # parScaler is AEmdnParScaler
    return nn_pred
    
    