# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:41:36 2020

@author: hpendyal
"""

import plotly.graph_objects as go
import numpy as np

from config import Config
config = Config()

def getPlotData(plotType, pdf_true_dict, pdf_pred_dict):
    if plotType == 'PUA':
        #--- yPred
        yPred = pdf_pred_dict['u'].mean(axis = 0)
        #--- bounds
        upperBound = pdf_pred_dict['u'].mean(axis=0) + pdf_pred_dict['u'].std(axis=0)
        lowerbound = pdf_pred_dict['u'].mean(axis=0) - pdf_pred_dict['u'].std(axis=0)
        
        return pdf_true_dict['u'], upperBound, yPred, lowerbound, config.pdfAbsoluteUp
    elif plotType == 'PDA':
        #--- yPred
        yPred = pdf_pred_dict['d'].mean(axis = 0)
        #--- bounds
        upperBound = pdf_pred_dict['d'].mean(axis=0) + pdf_pred_dict['d'].std(axis=0)
        lowerbound = pdf_pred_dict['d'].mean(axis=0) - pdf_pred_dict['d'].std(axis=0)
        
        return pdf_true_dict['d'], upperBound, yPred, lowerbound, config.pdfAbsoluteDown
    elif plotType == 'PUR':
        #--- PDF UP
        u_norm = pdf_true_dict['u']/pdf_true_dict['u'] 
        #--- bounds
        upperBound = pdf_pred_dict['u'].mean(axis=0)/pdf_true_dict['u'] + pdf_pred_dict['u'].std(axis=0)/pdf_true_dict['u']
        lowerbound = pdf_pred_dict['u'].mean(axis=0)/pdf_true_dict['u'] - pdf_pred_dict['u'].std(axis=0)/pdf_true_dict['u']                
        #--- yPred
        yPred = pdf_pred_dict['u'].mean(axis = 0)/pdf_true_dict['u']
        
        return u_norm,  upperBound, yPred, lowerbound, config.pdfRatioUp
    elif plotType == 'PDR':
        #--- PDF Down
        d_norm = pdf_true_dict['d']/pdf_true_dict['d']
        #--- bounds
        upperBound = pdf_pred_dict['d'].mean(axis=0)/pdf_true_dict['d'] + pdf_pred_dict['d'].std(axis=0)/pdf_true_dict['d']
        lowerbound = pdf_pred_dict['d'].mean(axis=0)/pdf_true_dict['d'] - pdf_pred_dict['d'].std(axis=0)/pdf_true_dict['d']       
        #--- yPred
        yPred = pdf_pred_dict['d'].mean(axis = 0)/pdf_true_dict['d']
        return d_norm,  upperBound, yPred, lowerbound, config.pdfRatioDown

#---- generate Plots Function
def generatePDFplots(x,yTrue,yPredUpper,yPred,yPredLower,graphTitle, yAxisType):
    #--- Pred ERROR trace upperBound
    predErrTraceUpper = go.Scatter(
                        x = x,
                        y = yPredUpper,
                        line_color='rgba(255,255,255,0)',
                        name='Pred Value Upper',
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        fillcolor='rgba(248, 148, 6, 0.4)',
                        fill='tonexty',
                        showlegend=False)

    #--- Pred trace
    predTrace = go.Scatter(x = x,
                         y = yPred,
                         #line_color='rgb(248, 148, 6)',
                         name='Pred Value',
                         mode='lines',
                         line=dict(color='rgba(217, 30, 24, 1)'),
                         #marker=dict(color="rgba(248, 148, 6,0.4)"),
                         fillcolor='rgba(248, 148, 6, 0.4)',
                         fill='tonexty',
                         showlegend=True)

    #--- Pred ERROR trace lowerBound
    predErrTraceLower = go.Scatter(
                        x = x,
                        y = yPredLower,
                        #fillcolor='rgba(248, 148, 6,0.4)',
                        #line_color='rgba(255,255,255,0)',
                        mode='lines',                    
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        name='Pred Value Lower',
                        showlegend=False)

    #--- concatenate the data
    predData = [predErrTraceLower, predTrace, predErrTraceUpper]
    predLayout = go.Layout(
                #yaxis=dict(title='Wind speed (m/s)'),
                xaxis_title="X",
                yaxis_type = 'linear' if yAxisType == 'lin' else 'log',
                title=dict(
                        text = graphTitle,
                        #xanchor = "left",
                        #yanchor = "top",
                        #pad = dict(l=50)
                    ),
                #showlegend = False
                )
    #--- create figure
    graphPlot = go.Figure(data=predData, layout=predLayout)

    #--- true trace
    True_Trace = go.Scattergl(
                        x = x,
                        y = yTrue,
                        name='True Value',
                        line_color='rgba(83, 51, 237, 1)',
                        showlegend=True)
    #--- add true trace - rgba(248, 148, 6, 1)
    graphPlot.add_trace(True_Trace)
    return graphPlot

