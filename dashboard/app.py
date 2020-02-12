# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:34:07 2020

@author: hpendyal
"""
#import all the libraries
import json
import numpy as np
import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


#--- load custom libraries.
from mlutils import backwardPredict
from mlutils import addNoiseSelected
from mlutils import calculate_xsec
from pdfutils import calculate_pdf

#--- initializing all the external properties
CNF_LOGO = "./assets/favicon.ico"
dataSelected = False


#-- Loading the data
X = np.load("data/X.npy")
Q2 = np.load("data/Q2.npy")
obs_p = np.load("data/obs_p.npy") #-- toy data is of shape (100,101)
obs_n = np.load("data/obs_n.npy") #-- toy data is of shape (100,101)

data = pd.DataFrame(data = X,
                    index=np.arange(len(X)),
                    columns=['X']
                    )
data['Q2'] = Q2
data['obs_p'] = obs_p[0,:] #--- currently taking only one sample
data['obs_n'] = obs_n[0,:] #--- currently taking only one sample
data['is_selected'] = False
data['co-ord'] = list(zip(X,Q2))

#---- calling the dash app
external_Scripts = ['https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js']                                           
mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'
app = dash.Dash(__name__,
                external_scripts = external_Scripts,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
               )
app.title = 'CNF Dashboard'
app.scripts.append_script({ 'external_url' : mathjax })



#--- navbar
navbar = html.Header(dbc.Navbar(
        [
                html.A(
                        dbc.Row([
                                dbc.Col(html.Img(src=CNF_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("CNF Dashboard", className="ml-2")),
                                ],
                                align='center',
                                no_gutters=True,
                                ),
                        href="#"                        
                        )                
                ],
        color='dark',
        dark=True,
        sticky="top",
))

#--- body
body = dbc.Container([
        dbc.Row([
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Parameters', className='h5-cnf'),
                                
                                #--- Uncertainity Column
                                html.Div(className='user-input-cnf',
                                    children=[
                                        html.Label('Uncertainity Value : '),
                                                                
                                        dcc.Input(id='ucert_value', type='text', value='0.05')]),
                                #--- Display Selected values
                                html.Div(children=[
                                        html.P('Selected Data'),
                                        #html.Pre(id='selected-data')
                                        dash_table.DataTable(id='selected-data',
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
                                        ]),
                                #--- Slider
                                html.Div(
                                    className='slider-box',
                                    children=[
                                    html.Div(children=[
                                    html.Label('F-value (Noise) :  '),
                                    dcc.Input(id='f-value-slider', type='text', value='')]),    
                                    html.Div(id='slider-output-container',
                                                 children=[
                                                     dcc.Slider(id='my-slider', min=0, max=5, step=0.0001, value=0, updatemode='drag', marks={0:{'label' : '0', 'style': {'color': '#f50'}},5:{'label': '5', 'style': {'color': '#f50'}}})
                                        ])
                                ]),
                                #--- Dropdown
                                html.Div(children=[
                                    html.Label('Select Model (Neural network Model): '),
                                    dcc.Dropdown(
                                            id='model-select',
                                            options=[
                                                    {'label':'--select--','value':'null'},
                                                    {'label': 'My Model', 'value': 'my_model'}
                                                    ],
                                            value='null'
                                            )]),
                                #--- Submit & Reset button
                                html.Div(children=[
                                    #--- Submit Button
                                    html.Button(id='submit-button', className='cnf-button', n_clicks=0, children='Submit'),
                                    #--- Reset Button
                                    html.Button(id='reset-button', className='cnf-button', n_clicks=0, children='Reset'),
                                    #--- output div
                                    
                                    ])
                                ]),
                        width=4,
                        ),
                #--- Input Graph
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Input Graph', className='h5-cnf'),
                                html.Div(id='input-graph-div',
                                         children=[
                                             dcc.Graph(id='g1',
                                                       figure = go.Figure(
                                                     data=go.Scattergl(
                                                         x=data['X'],
                                                         y=data['Q2'],
                                                         mode='markers',
                                                         ),
                                                     layout=go.Layout(
                                                         #title=go.layout.Title(text="Input Data Graph 1 (X,Q2)"),
                                                         xaxis_title="X",
                                                         yaxis_title="Q2",
                                                         )
                                                     )
                                                       )
                                                 ])
                                ]),
                        width=4,
                        ),
                #--- Output Graph
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Output Graph', className='h5-cnf'),
                                html.Div(id='output-graph-div',
                                         children=[
                                             dcc.Loading(
                                                 id='loading-output',
                                                 type='circle',
                                                 children=
                                                         dcc.Graph(id='output-graph-g2',
                                                             figure=go.Figure(
                                                                 data=[],
                                                                 layout=go.Layout(
                                                                     xaxis_title="Parameters",
                                                                     yaxis_title="Normalized values",
                                                                     )
                                                                 ),
                                                             animate=True)
                                                         )
                                             ])
                                ]),
                        width=4,
                        ),
                ])
        ],
        fluid=True)


                                                                
                                                                

                                                                

#--- server app
server = app.server
                                                                
#--- Final Layout
app.layout = html.Div(children=[
        navbar,
        body,
])

#--- display selected points
@app.callback(
    [Output('selected-data', 'data'),
     Output('selected-data', 'columns')
     ],
    [Input('g1', 'selectedData')])
def display_selected_data(selectedData):
    #--- parsing the selected data
    s_data = json.dumps(selectedData, indent=2)
    if s_data != 'null':
        #--- set the data_Selected falg to true
        dataSelected = True
        selected = json.loads(s_data)
        temp_x = []
        temp_q2 = []
        temp_op = []
        temp_on = []
        for point in selected['points']:
            #return point
            #--- update the is_selected column
            data.loc[data['co-ord'] == (point['x'],point['y']),'is_selected'] =  True
            temp_op.append(data.loc[data['co-ord'] == (point['x'],point['y']), 'obs_p'])
            temp_on.append(data.loc[data['co-ord'] == (point['x'],point['y']), 'obs_n'])
            temp_x.append(point['x'])
            temp_q2.append(point['y'])
        #points = s_data['points']
        df = pd.DataFrame(data=temp_x, columns=["X"])
        df['Q2'] = temp_q2
        df['obs_p'] = temp_op
        df['obs_n'] = temp_on
        #{
         #       'x' : temp_x,
          #      'y' : temp_y 
           #     }
        return df.to_dict('records'),[{"name":i,"id":i} for i in df.columns]
    return [{}],[]

#--- slider callback
@app.callback(
    Output('f-value-slider', 'value'),
    [Input('my-slider', 'value')])
def update_slider(value):
    return format(value)


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

#--- callback for Reset to clear dropdown
@app.callback(
            Output('model-select', 'value'),
        [Input('reset-button', 'n_clicks')])
def reset_data_dropdown(n_clicks):
    #--- make sure the data['is_selected'] is returned to false
    data['is_selected'] = False
    return ""

#--- callback for Reset to refresh the grap


#--- callback after submit
@app.callback(
        Output('output-graph-g2', 'figure'),
        [Input('submit-button', 'n_clicks')],
        [
         State('selected-data', 'data'),
         State('ucert_value','value'),
         State('model-select','value'),
         State('f-value-slider', 'value')])
def update_output(n_clicks, tabledata, uncertainity_value, model_select, f_value):
    # --- update the values
    if tabledata is not None:
        if model_select != 'null':
            #data_new = pd.DataFrame(tabledata)
            #--- udpate the dataframe and display all the columns
            
            #-- file name to be saved
            fname = 'test_backward'
            
            #-- check if No data is selected apply noise to every column
            if not dataSelected :
                data['is_selected'] = True
                print('--> (RUNLOG) - No Data is selected, so applying noise to all columns.')
            #--- add uncertainity column to the database
            #data['uncertainity'] = float(uncertainity_value)
            #--- get the list of data with added noise
            obs_p_noised = addNoiseSelected(data, f_value, uncertainity_value, 'is_selected', 'obs_p', len(data['obs_p']))
            obs_n_noised = addNoiseSelected(data, f_value, uncertainity_value, 'is_selected', 'obs_n', len(data['obs_n']))
            
             
            #--- concatenate the lists to get the cross-sectional Data
            xsec = calculate_xsec(obs_p_noised, obs_n_noised)
            nn_pred = backwardPredict(fname, model_select, xsec)
            
            #--- use the predicted values to get the pdf-up and pdf-down values
            pdf_dict = calculate_pdf(nn_pred)
            
            #--- Plot the figure
            figure_g = go.Figure()
            pdf_up_trace = go.Scatter(
                        x = pdf_dict['x-axis'],
                        y = pdf_dict['u'].mean(axis=0),
                        name='PDF UP',
                        showlegend=True,
                        error_y=dict(
                                type='data',
                                color='orange',
                                array=pdf_dict['u'].std(axis=0)*5,
                                visible=True
                                )
                        )
            #--- Plotting pdf_up
            figure_g.add_trace(pdf_up_trace)
            pdf_down_trace = go.Scatter(
                        x = pdf_dict['x-axis'],
                        y = pdf_dict['d'].mean(axis=0),
                        name='PDF DOWN',
                        showlegend=True,
                        error_y=dict(
                                type='data',
                                color='yellow',
                                array=pdf_dict['d'].std(axis=0)*5,
                                visible=True
                            )
                        )
            #--- Plotting pdf_down
            figure_g.add_trace(pdf_down_trace)
            #-- add a dropdown to scale log and linear
            figure_g.update_layout(
                #title='Predicted parameters with '+str(f_value)+' noise',
                xaxis_title="X",
                #yaxis_title="Y",
                #annotations=[dict(text='Scale : ', showarrow=False, x=0, y=1.10, yref='paper', align='left')],
                updatemenus=list([
                        dict(type="buttons",
                             active=1,
                             direction="right",
                             pad={"r": 10, "t": 10},
                             showactive=True,
                             x=0,
                             xanchor="left",
                             y=1.20,
                             yanchor="top",
                             buttons=list([
                                     dict(label='Log',
                                          method='update',
                                          args=[{'visible': [True, True]},
                                                 {'title': 'Log scale',
                                                  'yaxis': {'type': 'log'}}]),
                                    dict(label='Linear',
                                         method='update',
                                         args=[{'visible': [True, True]},
                                                {'title': 'Linear scale',
                                                 'yaxis': {'type': 'linear'}}])
                            ]),
                        )
                    ])
                )
            return figure_g
        else:
            return go.Figure(
             data=[],
             layout=go.Layout(
                 title = 'Output Graph No Model Selected',
                 xaxis_title="Parameters",
                 yaxis_title="Normalized values",
                 )
             )#--- throw exception saying select model
    else:
         return go.Figure(
             data=[],
             layout=go.Layout(
                 title = 'Output Graph with no Data',
                 xaxis_title="Parameters",
                 yaxis_title="Normalized values",
                 )
             )

# go.Figure(data=[])
#--- callback to update the graph title based on the noise
#@app.callback(Output('output-graph-g2', 'figure'),
#              [Input('output-graph-g2', 'figure')],
#              [State('f-value-slider', 'value')])
#def update_graph_title(fig, f_value):
#    fig.update_layout(
#        title=go.layout.Title(text='Predicted parameters with '+f_value+' noise'))
#    return fig


if __name__ == '__main__':
    app.run_server(debug = False)
    


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