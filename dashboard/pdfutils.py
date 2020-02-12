# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:01:43 2020

@author: hpendyal
"""

import sys,os
import numpy as np
from scipy.special import gamma
from tools import save, load,checkdir,lprint

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