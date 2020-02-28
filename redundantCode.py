# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:22:15 2020

@author: hpendyal
"""

#data['err_obs_p'] = data['obs_p'].copy()
##data['err_obs_n'] = data['obs_n'].copy()
#data['err_obs_p_mod'] = data['err_obs_p'].copy()
#data['err_obs_n_mod'] = data['err_obs_n'].copy()
#data['co-ord'] = list(zip(X,Q2))

# go.Figure(data=[])
#--- callback to update the graph title based on the noise
#@app.callback(Output('output-graph-g2', 'figure'),
#              [Input('output-graph-g2', 'figure')],
#              [State('f-value-slider', 'value')])
#def update_graph_title(fig, f_value):
#    fig.update_layout(
#        title=go.layout.Title(text='Predicted parameters with '+f_value+' noise'))
#    return fig

#--- add uncertainity column to the database
            #data['uncertainity'] = float(uncertainity_value)
            #--- get the list of data with added noise
            #obs_p_noised = addNoiseSelected(data, f_value, uncertainity_value, 'is_selected', 'obs_p', len(data['obs_p']))
            #obs_n_noised = addNoiseSelected(data, f_value, uncertainity_value, 'is_selected', 'obs_n', len(data['obs_n']))
            
            #--- concatenate the lists to get the cross-sectional Data
            #xsec = calculate_xsec(obs_p_noised, obs_n_noised)


#--- generate figure
            #figure_g = generatePDFplots(pdf_true_dict['x-axis'], yTrue, upperBound, yPred, lowerBound, plotTitle, scale_value)

            #--- Plot the figure
            #figure_g = go.Figure()
            #pdf_up_trace = go.Scatter(
               #            x = pdf_dict['x-axis'],
                #            y = np.ones(len(pdf_dict['x-axis'])),#pdf_dict['u'].mean(axis=0),
                #            name='PDF UP',
                #            showlegend=True,
                #            error_y=dict(
                #                    type='data',
                #                    color='orange',
                #                    array=pdf_dict['u'].std(axis=0)/pdf_dict['d'].mean(axis=0),
                #                    #array=pdf_dict['u'].std(axis=0)*5,
                #                    visible=True
                #                    )
                #            )
                #--- Plotting pdf_up
                #figure_g.add_trace(pdf_up_trace)
                #pdf_down_trace = go.Scatter(
                #            x = pdf_dict['x-axis'],
                #            y = pdf_dict['d'].mean(axis=0),
                #            name='PDF DOWN',
                #            showlegend=True,
                #           error_y=dict(
                #                    type='data',
                #                    color='yellow',
                #                    array=pdf_dict['d'].std(axis=0)*5,
                #                    visible=True
                #                )
                #            )
                #--- Plotting pdf_down
                #figure_g.add_trace(pdf_down_trace)
    
                
                #-- add a dropdown to scale log and linear
                #figure_g.update_layout(
                    #title='Predicted parameters with '+str(f_value)+' noise',
                    #xaxis_title="X",
                    #yaxis_title="Y",
                    #annotations=[dict(text='Scale : ', showarrow=False, x=0, y=1.10, yref='paper', align='left')],
                #    updatemenus=list([
                #            dict(type="buttons",
                #                 active=1,
                #                 direction="right",
                #                 pad={"r": 10, "t": 10},
                #                 showactive=True,
                #                 x=0,
                #                 xanchor="left",
                #                 y=1.20,
                #                 yanchor="top",
                #                 buttons=list([
                #                         dict(label='Log',
                #                              method='update',
                #                              args=[{'visible': [True, True]},
                #                                     {'title': 'Log scale',
                #                                      'yaxis': {'type': 'log'}}]),
                #                        dict(label='Linear',
                #                             method='update',
                #                             args=[{'visible': [True, True]},
                #                                    {'title': 'Linear scale',
                #                                     'yaxis': {'type': 'linear'}}])
                #                ]),
                #            )
                #        ])
                #    )

#--- callback for Reset to clear datatable
'''
@app.callback(
            [Output('selected-data', 'data'),
            Output('selected-data', 'columns')],
            [Input('reset-button', 'n_clicks')])
def reset_dataTable(n_clicks):
    #--- make sure the data['is_selected'] is returned to false
    data['is_selected'] = False
    return [{}],[]
'''

#----- Redundant Data ----#
'''
data['obs_p-N'] = np.where(data['is_selected']==True, data['obs_p'] + (float(f_value) * np.random.rand(num_features,)), data['obs_p'])
data['obs_n-N'] = np.where(data['is_selected']==True, data['obs_n'] + (float(f_value) * np.random.rand(num_features,)), data['obs_n'])
print('-------'+str(data_new["obs_p"][0])+'--------------')
data_new['obs_p-N'] = [float(x[0]) + (float(f_value) * np.random.rand(num_features,)) for x in data_new['obs_p']]
data_new['obs_n-N'] = [float(x[0]) + (float(f_value) * np.random.rand(num_features,)) for x in data_new['obs_n']]
data_new['is_selected'] = False
data_return = data_new[['X','Q2','obs_p', 'obs_n', 'obs_p-N', 'obs_n-N']]
#return str(data_new)
return data_return.to_dict('records'),[{"name":i,"id":i} for i in data_return]
'''

'''
#data['obs_p-N'] = np.where(data['is_selected']==True, data['obs_p'] + (float(f_value) * np.random.rand(num_features,)), data['obs_p'])
            #data['obs_n-N'] = np.where(data['is_selected']==True, data['obs_n'] + (float(f_value) * np.random.rand(num_features,)), data['obs_n'])
            #print('-------'+str(data_new["obs_p"][0])+'--------------')
            #data_new['obs_p-N'] = [float(x[0]) * float(f_value) for x in data_new['obs_p']]
            #data_new['obs_n-N'] = [float(x[0]) * float(f_value) for x in data_new['obs_n']]
            #data_new['is_selected'] = False
            #data_return = data_new[['X','Q2','obs_p', 'obs_n', 'obs_p-N', 'obs_n-N']]
            #return str(data_new)
            #return data_return.to_dict('records'),[{"name":i,"id":i} for i in data_return.columns]
'''

'''
#--- callback to update the output graph
@app.callback(Output('output-graph-g2', 'figure'),
              [Input('output-state', 'data')],
              [State('model-select','value'),
               State('f-value-slider', 'value')])
def update_graph_scatter(updated_data, model_select, f_value):
    if updated_data is not None:
        if model_select != 'null':
            fname = 'test_backward'
            #modelname = 'model_1'
            #datadict = eval(updated_data)
            
            nn_pred = backwardPredict(fname, model_select, float(f_value), data)
            figure_g = go.Figure(
                data = go.Scatter(
                    x=np.arange(10),
                    y=nn_pred.mean(axis=0),
                    name='pred mean',
                    showlegend = True,
                    error_y=dict(
                            type='data', # value of error bar given in data coordinates,
                            color='orange',
                            array=nn_pred.std(axis = 0)*5,
                            visible=True
                            )
                    )
                )
            #(data, ['X0','y_new'], model_select)
            figure_g.update_layout(
                #title='Predicted parameters with '+str(f_value)+' noise',
                xaxis_title="Parameters",
                yaxis_title="Normalized values",
                )
            #layout_g = go.Layout({'title':'Predicted parameters with '+f_value+' noise', 'xaxis':{'title':'Parameters'}, 'yaxis':{'title':'Normalized values'}})
            return figure_g
        else:
            return {'data': [], 'layout' : go.Layout({'title':'Output Graph No Model Selected', 'xaxis':{'title':'Parameters'}, 'yaxis':{'title':'Normalized values'}})} #--- throw exception saying select model
    else:
        return {'data' : [], 'layout' : go.Layout({'title':'Output Graph with no Data', 'xaxis':{'title':'Parameters'}, 'yaxis':{'title':'Normalized values'}})}
'''

'''
Output- Div

html.Div(children=[
                                            html.P('Output Data '),
                                            #html.Pre(id='output-state'),
                                            dash_table.DataTable(id='output-state',
                                                             fixed_rows={ 'headers': True, 'data': 0 },
                                                             style_table={
                                                                     'height': '200px',
                                                                     #'overflowY': 'scroll',
                                                                     },
                                                             style_cell={
                                                                 'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                                                                 'overflow': 'hidden',
                                                                 'textOverflow': 'ellipsis',
                                                                 })
                                            ])
'''

'''
            dataplot = go.Scatter(
                    x=np.arange(10),
                    y=nn_pred.mean(axis=0),
                    name='pred mean',
                    showlegend = True,
                    error_y=dict(
                            type='data', # value of error bar given in data coordinates,
                            color='orange',
                            array=nn_pred.std(axis = 0)*5,
                            visible=True
                            )
                    )
            #--- predicted Output
            #--- true output toy data
            par = np.load("./data/par.npy") # this is the output
            figure_g.add_trace(go.Scatter(
                    x=np.arange(10),
                    y=par[0],
                    name='True Value'
                    ))
'''