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


#--- initializing all the external things
CNF_LOGO = "./assets/favicon.ico"
num_features = 101



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


#app = dash.Dash()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                                                     dcc.Slider(id='my-slider', min=0, max=1, step=0.0001, value=0, updatemode='drag', marks={0:{'label' : '0', 'style': {'color': '#f50'}},1:{'label': '1', 'style': {'color': '#f50'}}})
                                        ])
                                ]),
                                #--- Dropdown
                                html.Div(children=[
                                    html.Label('Select Model (Neural network Model): '),
                                    dcc.Dropdown(
                                            id='model-select',
                                            options=[
                                                    {'label':'--select--','value':'null'},
                                                    {'label': 'Model 1', 'value': 'model_1'},
                                                    {'label': 'My Model', 'value': 'my_model'}
                                                    ],
                                            value='null'
                                            )]),
                                #--- Submit button
                                html.Div(children=[
                                    #--- Submit Button
                                    html.Button(id='submit-button', n_clicks=0, children='Submit'),
                                    #--- Reset Button
                                    html.Button(id='reset-button', n_clicks=0, children='Reset'),
                                    #--- output div
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
                                    ])
                                ]),
                        width=3,
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
                        ),
                #--- Output Graph
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Output Graph', className='h5-cnf'),
                                html.Div(id='output-graph-div',
                                         children=[dcc.Graph(id='output-graph-g2',
                                                             figure=go.Figure(
                                                                 data=[],
                                                                 layout=go.Layout(
                                                                     xaxis_title="Parameters",
                                                                     yaxis_title="Normalized values",
                                                                     )
                                                                 ),
                                                             animate=True)])
                                ]),
                        ),
                ])
        ],
        fluid=True)


#--- Final Layout
app.layout = html.Div(children=[
        navbar,
        body,
])

#--- display points
@app.callback(
    [Output('selected-data', 'data'),
     Output('selected-data', 'columns')],
    [Input('g1', 'selectedData')])
def display_selected_data(selectedData):
    #--- parsing the selected data
    s_data = json.dumps(selectedData, indent=2)
    if s_data != 'null':
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

#--- callback after submit
@app.callback(
        [Output('output-state', 'data'),
         Output('output-state', 'columns')],
        [Input('submit-button', 'n_clicks')],
        [
         State('selected-data', 'data'),
         #State('selected-data', 'children'),
         State('f-value-slider', 'value')])
def update_output(n_clicks, str_dict, f_value):
    # --- update the values
    if str_dict is not None:
        data_new = pd.DataFrame(str_dict)
        #--- udpate the dataframe and displaying all the columns
        data['obs_p-N'] = np.where(data['is_selected']==True, data['obs_p'] + (float(f_value) * np.random.rand(num_features,)), data['obs_p'])
        data['obs_n-N'] = np.where(data['is_selected']==True, data['obs_n'] + (float(f_value) * np.random.rand(num_features,)), data['obs_n'])
        #print('-------'+str(data_new["obs_p"][0])+'--------------')
        data_new['obs_p-N'] = [float(x[0]) * float(f_value) for x in data_new['obs_p']]
        data_new['obs_n-N'] = [float(x[0]) * float(f_value) for x in data_new['obs_n']]
        data_new['is_selected'] = False
        data_return = data_new[['X','Q2','obs_p', 'obs_n', 'obs_p-N', 'obs_n-N']]
        #return str(data_new)
        return data_return.to_dict('records'),[{"name":i,"id":i} for i in data_return.columns]
        #return [{}],[]
    return [{}],[]

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