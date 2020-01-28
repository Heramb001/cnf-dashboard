# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go

#--- custom utils
from utils import cluster_data

data_b = pd.read_csv('./data/B.txt', header=None, prefix = 'X', delimiter=' ')
data_b['is_selected'] = False
data_b['co-ord'] = list(zip(data_b['X0'],data_b['X1']))


#app = dash.Dash()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#--- navbar
navbar = dbc.NavbarSimple(
    brand="GAN Dashboard",
    brand_href="#",
    sticky="top",
)

#--- body
body = dbc.Container([
        dbc.Row([
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Hyper-Parameters'),
                                #--- Display Selected values
                                html.Div(children=[
                                        html.P('Selected Data'),
                                        html.Pre(id='selected-data')]),
                                #--- Slider
                                html.Div(children=[
                                        html.Div(id='slider-output-container',
                                                 children=[html.Label('F-value : '),
                                                  dcc.Input(id='f-value-slider', type='text', value='')]),
                                                  dcc.Slider(id='my-slider', min=0, max=1, step=0.0001, value=0, updatemode='drag', marks={0:{'label' : '0', 'style': {'color': '#77b0b1'}},1:{'label': '1', 'style': {'color': '#f50'}}})
                                        ]),
                                #--- Dropdown
                                html.Div(children=[
                                    html.Label('Select Model (for calculating cluster similarity): '),
                                    dcc.Dropdown(
                                            id='model-select',
                                            options=[
                                                    {'label':'--select--','value':'null'},
                                                    {'label': 'MINIMUM', 'value': 'single'},
                                                    {'label': 'MAXIMUM', 'value': 'complete'},
                                                    {'label': 'Group Average', 'value': 'average'},
                                                    {'label': 'Centroid Distance', 'value': 'ward'}
                                                    ],
                                            value='null'
                                            )]),
                                #--- Submit button
                                html.Div(children=[
                                    #--- Submit
                                    html.Button(id='submit-button', n_clicks=0, children='Submit'),
                                    #--- output div
                                    html.Div(children=[
                                            html.P('Output Data '),
                                            html.Pre(id='output-state'),
                                            ])
                                    ])
                                ]),
                        width=3,
                        ),
                #--- Input Graph
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Input Graph'),
                                html.Div(id='input-graph-div',
                                         children=[
                                                 dcc.Graph(id='g1',
                                                           figure={
                                                 'data':[{'x':data_b['X0'],'y':data_b['X1'], 'mode':'markers', 'name':'data 1', 'marker': {'size': 8}},
                                                               ],
                                                 'layout':{
                                                              'title':'Input Graph',
                                                              'xaxis':{'title':'X'},
                                                              'yaxis':{'title':'Q2'},
                                                              'clickmode': 'event+select'
                                                          }
                                                      })
                                                 ])
                                ]),
                        ),
                #--- Output Graph
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('Output Graph'),
                                html.Div(id='output-graph-div',
                                         children=[dcc.Graph(id='output-graph-g2', animate=True)])
                                ]),
                        ),
                ])
        ],
        fluid=True)

app.layout = html.Div(children=[
        navbar,
        body,
])


#--- display points
@app.callback(
    Output('selected-data', 'children'),
    [Input('g1', 'selectedData')])
def display_selected_data(selectedData):
    #--- parsing the selected data
    s_data = json.dumps(selectedData, indent=2)
    if s_data != 'null':
        selected = json.loads(s_data)
        temp_x = []
        temp_y = []
        for point in selected['points']:
            #return point
            #--- update the is_selected column
            data_b.loc[data_b['co-ord'] == (point['x'],point['y']),'is_selected'] =  True
            temp_x.append(point['x'])
            temp_y.append(point['y'])
        #points = s_data['points']
        pointsdict = {
                'x' : temp_x,
                'y' : temp_y 
                }
        
        return str(pointsdict)
    return s_data

#--- slider callback
@app.callback(
    Output('f-value-slider', 'value'),
    [Input('my-slider', 'value')])
def update_slider(value):
    return format(value)

#--- callback after submit
@app.callback(
        Output('output-state', 'children'),
        [Input('submit-button', 'n_clicks')],
        [State('selected-data', 'children'),
         State('f-value-slider', 'value')])
def update_output(n_clicks, str_dict, f_value):
    # --- update the values
    if str_dict is not None:
        data = eval(str_dict)
        data_b['y_new'] = np.where(data_b['is_selected']==True, data_b['X1']+float(f_value), data_b['X1'])
        data['y-new'] = [float(x) + float(f_value) for x in data['y']]
        data['is_selected'] = False
        return str(data)
    return None

#--- callback to update the output graph
@app.callback(Output('output-graph-g2', 'figure'),
              [Input('output-state', 'children')],
              [State('model-select','value')])
def update_graph_scatter(updated_data, model_select):
    if updated_data is not None:
        if model_select != 'null':
            datadict = eval(updated_data)
            clusters = cluster_data(data_b, ['X0','y_new'], model_select)
            data = go.Scatter(
                    x=data_b['X0'],
                    y=data_b['y_new'],
                    name='Scatter',
                    showlegend = True,
                    marker={
                            'color':clusters['result']
                            },
                    mode= 'markers'
                    )
            return {'data': [data],'layout' : go.Layout({'title':'Output Graph'})}
        else:
            return {'data': [], 'layout' : go.Layout({'title':'Output Graph No Model Selected'})} #--- throw exception saying select model
    else:
        return {'data' : [], 'layout' : go.Layout({'title':'Output Graph with no Data'})}


if __name__ == '__main__':
    app.run_server(debug = False)