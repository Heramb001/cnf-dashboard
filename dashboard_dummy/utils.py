#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 13:59:54 2020

@author: heramb
"""
import numpy as np                          #--- Numpy for numeric calculations
import pandas as pd                         #--- For Data Representation and Analysis
# from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering 


def cluster_data(data, columns, similarity_link):
    if similarity_link == 'MINIMUM':
        model = AgglomerativeClustering(n_clusters = 2, linkage='single')
        result = model.fit_predict(data[columns])
        data['result'] = result
        return data
        return result
    elif similarity_link == 'MAXIMUM':
        model = AgglomerativeClustering(n_clusters = 2, linkage='complete')
        result = model.fit_predict(data[columns])
        data['result'] = result
        return data
    elif similarity_link == 'Group Average':
        model = AgglomerativeClustering(n_clusters = 2, linkage='average')
        result = model.fit_predict(data[columns])
        data['result'] = result
        return data
    elif similarity_link == 'Centroid Distance':
        model = AgglomerativeClustering(n_clusters = 2, linkage='ward')
        result = model.fit_predict(data[columns])
        data['result'] = result
        return data
    else:
        model = AgglomerativeClustering(n_clusters = 2, linkage='single')
        result = model.fit_predict(data[columns])
        data['result'] = result
        return data #--- if nothing is specified run single linkage on data
        
    
    