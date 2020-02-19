# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:01:43 2020

@author: hpendyal
"""

import sys,os
import numpy as np
from scipy.special import gamma

Q02=4.0
lam2=0.2**2
pu=np.ones(10)
pd=np.ones(10)
pu[5:]*=0.1
pd[5:]*=0.1

def set_params(par):
    pu[:5]=par[:5]
    pd[:5]=par[5:]
  
def get_s(Q2):
    return np.log(np.log(Q2/lam2)/np.log(Q02/lam2))

def _get_shape(x,p):
    return p[0]*x**p[1]*(1-x)**p[2]*(1+p[3]*x+p[4]*x**2)
    
def get_shape(x,p,s):
    N=p[0] + p[5] * s
    a=p[1] + p[6] * s
    b=p[2] + p[7] * s
    c=p[3] + p[8] * s
    d=p[4] + p[9] * s
    return _get_shape(x,[N,a,b,c,d])

def get_pdf(x,Q2,flav):
    s=get_s(Q2)
    if flav=='u': return get_shape(x,pu,s)
    if flav=='d': return get_shape(x,pd,s)
    
def calculate_pdf(data_par, pred = True):
    x = np.linspace(0.01,0.99,100)
    Q2=4.0
    u=[]
    d=[]
    if pred:                    #--- For True, we are calculating pdf for Predicted values
        for i in range(data_par.shape[0]):
            set_params(data_par[i])
            u.append(get_pdf(x,Q2,'u'))
            d.append(get_pdf(x,Q2,'d'))
    else:                        #--- For False, we are calculating pdf for Ground Truth
       set_params(data_par)
       u = get_pdf(x,Q2,'u')
       d = get_pdf(x,Q2,'d')       
    u = np.array(u)
    d = np.array(d)
    print('--> (RUNLOG) - Up data shape : ',u.shape)
    print('--> (RUNLOG) - Down data Shape : ',d.shape)
    return {'u' : u, 'd' : d, 'x-axis' : x}