# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:37:58 2020

@author: hpendyal
"""

class Config:
    def __init__(self,
                 #--- Logo of dashboard
                 CNF_LOGO = "./assets/favicon.ico",
                 TITLE = "CNF Dashboard",
                 #--- external URLs
                 MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML",
                 #--- Sample To be selected
                 DEFAULT_SAMPLE = 0,
                 #--- Uncertainity Values
                 DEFAULT_UNCERTAINITY = 0.05,
                 #--- Noise Values
                 DEFAULT_NOISE = 0.5,
                 NOISE_MIN = 0,
                 NOISE_MAX = 5,
                 #--- paths of data files
                 xPath = "data/X.npy",
                 q2path = "data/Q2.npy",
                 obspPath = "data/obs_p.npy",
                 obsnPath = "data/obs_n.npy",
                 parPath = "data/par.npy",
                 #--- Plot titles
                 pdfAbsoluteUp = 'PDF-Up Absolute Plot',
                 pdfAbsoluteDown = 'PDF-Down Absolute Plot',
                 pdfRatioUp = 'PDF-Up Ratio Plot',
                 pdfRatioDown = 'PDF-Down Ratio Plot',
                 ):
        #--- Logo of dashboard
        self.__CNF_LOGO = CNF_LOGO
        self.__TITLE = TITLE
        #--- external URLs
        self.__MATHJAX = MATHJAX
        #--- Sample To be selected
        self.__DEFAULT_SAMPLE = DEFAULT_SAMPLE
        #--- Uncertainity Values
        self.__DEFAULT_UNCERTAINITY = DEFAULT_UNCERTAINITY
        #--- Noise Values
        self.__DEFAULT_NOISE = DEFAULT_NOISE
        self.__NOISE_MIN = NOISE_MIN
        self.__NOISE_MAX = NOISE_MAX
        #--- paths of data files
        self.__xPath = xPath
        self.__q2path = q2path
        self.__obspPath = obspPath
        self.__obsnPath = obsnPath
        self.__parPath = parPath
        #--- Plot titles
        self.__pdfAbsoluteUp = pdfAbsoluteUp
        self.__pdfAbsoluteDown = pdfAbsoluteDown
        self.__pdfRatioUp = pdfRatioUp
        self.__pdfRatioDown = pdfRatioDown
        
    #--- getters
    @property
    def CNF_LOGO(self):
        return self.__CNF_LOGO
    
    @property
    def TITLE(self):
        return self.__TITLE
    
    @property
    def MATHJAX(self):
        return self.__MATHJAX

    @property
    def DEFAULT_SAMPLE(self):
        return self.__DEFAULT_SAMPLE
    
    @property
    def DEFAULT_UNCERTAINITY(self):
        return self.__DEFAULT_UNCERTAINITY
    
    @property
    def DEFAULT_NOISE(self):
        return self.__DEFAULT_NOISE
    
    @property
    def NOISE_MIN(self):
        return self.__NOISE_MIN
    
    @property
    def NOISE_MAX(self):
        return self.__NOISE_MAX
    
    @property
    def xPath(self):
        return self.__xPath
    
    @property
    def q2path(self):
        return self.__q2path
    
    @property
    def obspPath(self):
        return self.__obspPath
    
    @property
    def obsnPath(self):
        return self.__obsnPath
    
    @property
    def parPath(self):
        return self.__parPath
    
    @property
    def pdfAbsoluteUp(self):
        return self.__pdfAbsoluteUp
    
    @property
    def pdfAbsoluteDown(self):
        return self.__pdfAbsoluteDown
    
    @property
    def pdfRatioUp(self):
        return self.__pdfRatioUp
    
    @property
    def pdfRatioDown(self):
        return self.__pdfRatioDown
    
    #--- setters
    def setDEFAULT_SAMPLE(self, val):
            self.__DEFAULT_SAMPLE = val
            
    def setNOISE_MIN(self, val):
            self.__NOISE_MIN = val
            
    def setNOISE_MAX(self, val):
            self.__NOISE_MAX = val