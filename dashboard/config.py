# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:37:58 2020

@author: hpendyal
"""

#--- Logo of dashboard
CNF_LOGO = "./assets/favicon.ico"
TITLE = "CNF Dashboard"

#--- external URLs
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"

#--- Uncertainity Values
DEFAULT_UNCERTAINITY = 0.05

#--- Noise Values
DEFAULT_NOISE = 0.5
NOISE_MIN = 0
NOISE_MAX = 5

#--- paths of data files
xPath = "data/X.npy"
q2path = "data/Q2.npy"
obspPath = "data/obs_p.npy"
obsnPath = "data/obs_n.npy"
parPath = "data/par.npy"

#--- Path to ML files

#--- Plot titles
pdfAbsoluteUp = 'PDF-Up Absolute Plot'
pdfAbsoluteDown = 'PDF-Down Absolute Plot'
pdfRatioUp = 'PDF-Up Ratio Plot'
pdfRatioDown = 'PDF-Down Ratio Plot'
