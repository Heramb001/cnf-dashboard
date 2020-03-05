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
from utils.scalerUtils import scaleCross
from utils.mlutils import backwardPredict
from utils.mlutils import addNoise
from utils.pdfutils import calculate_pdf
from utils.graphutils import generatePDFplots, getPlotData



#--- import config to get all external dependencies
from config import Config
#--- create Config Class
config = Config()

#--- initializing all the external properties
CNF_LOGO = config.CNF_LOGO
dataSelected = False


#-- Loading the data
X = np.load(config.xPath)
Q2 = np.load(config.q2path)
#obs_p = np.load(config.obspPath) #-- toy data is of shape (100,101)
#obs_n = np.load(config.obsnPath) #-- toy data is of shape (100,101)
xsecData = np.load(config.xsecPath) #--- toy data of shape()

#---True Data
trueBar = np.load(config.parPath)


data = pd.DataFrame(data = X,
                    index=np.arange(len(X)),
                    columns=['X']
                    )
data['Q2'] = Q2
#data['obs_p'] = obs_p[config.DEFAULT_SAMPLE,:] #--- currently taking only one sample
#data['obs_n'] = obs_n[config.DEFAULT_SAMPLE,:] #--- currently taking only one sample
data['is_selected'] = False

#---- calling the dash app
external_Scripts = ['https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js']                                           
mathjax = config.MATHJAX

app = dash.Dash(__name__,
                external_scripts = external_Scripts,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
               )
app.title = config.TITLE
app.scripts.append_script({ 'external_url' : mathjax })





#--- Config button
config_button = dbc.Row([
            dbc.Col(
                html.Div([
                    dbc.Button("Config", id="open", className="ml-2", color="warning"),
                    dbc.Modal([
                dbc.ModalHeader("Update Config"),
                dbc.ModalBody([
                    #--- defualtSample to be selected
                    html.Div([
                        html.Label('Default Sample :  '),
                        dcc.Input(id='updateSample', type='text', value=str(config.DEFAULT_SAMPLE))
                        ]),
                    #--- Noise Min and Noise Max
                    html.Div([
                            #--- Noise Min
                            html.Div([
                                html.Label('Noise Min :  '),
                                dcc.Input(id='updateNoiseMin', type='text', value=str(config.NOISE_MIN))
                                ]),
                            #--- Noise Max
                            html.Div([
                                html.Label('Noise Max :  '),
                                dcc.Input(id='updateNoiseMax', type='text', value=str(config.NOISE_MAX))
                                ]),
                            ],
                        style={'display': 'inline-block'}                       
                        )
                    ]),
                dbc.ModalFooter([
                    html.Div([
                        dbc.Button("Update", id="update-config", color="primary", className="mr-2"),
                        dbc.Button("Close", id="close", className="mr-2")
                        ])
                ]),
            ],
            id="modal",
            size="lg",
        ),]),
                width="auto",
                ),
            ],
        no_gutters=True,
        className="ml-auto flex-nowrap mt-3 mt-md-0",
        align="center",
        )

#--- navbar
navbar = html.Header(dbc.Navbar(
        [
                html.A(
                        dbc.Row([
                                dbc.Col(html.Img(src=CNF_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand(config.TITLE, className="ml-2")),
                                ],
                                align='center',
                                no_gutters=True,
                                ),
                        href="#"                        
                        ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(config_button, id="navbar-collapse", navbar=True),                
                ],
        color='dark',
        dark=True,
        sticky="top",
))

#--- body
body = dbc.Container([
        dbc.Row([
                #--- Input Graph
                dbc.Col([
                        html.Div(children=[
                                #--- Label
                                html.H5('DIS Kinematics', className='h5-cnf'),
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
                                                         yaxis_title="Q<sup>2</sup> (GeV<sup>2</sup>)",
                                                         )
                                                     )
                                                       )
                                                 ]),
                                html.Div([                                  
                                    ]),
                                ]),
                        html.Div(children=[
                                #--- Label
                                html.H5('Setup', className='h5-cnf'),
                                #--- Select Data Sample
                                html.Div([
                                        html.Label(['Select Data sample']),
                                        dcc.Input(id='xsec-data', type='number', min=0, max=len(xsecData)-1,value=config.DEFAULT_SAMPLE, style={'margin-left':'8px','margin-right':'8px'}),
                                        ]),
                                #--- Uncertainity Column
                                html.Div(className='user-input-cnf',
                                    children=[
                                        html.Label('Default Relative Uncertainity '),
                                        dcc.Input(id='ucert_value', type='text', value=str(config.DEFAULT_UNCERTAINITY), style={'margin-left':'8px'})]),
                                #--- Slider
                                html.Div(
                                    className='slider-box',
                                    children=[
                                    html.Div(children=[
                                    html.Label('Uncertainity Rescaling Factor '),
                                    dcc.Input(id='f-value-slider', type='text', value=str(config.DEFAULT_NOISE), style={'margin-left':'8px'})]),    
                                    html.Div(id='slider-output-container',
                                                 children=[
                                                     dcc.Slider(id='my-slider', min=config.NOISE_MIN, max=config.NOISE_MAX, step=0.0001, value=config.DEFAULT_NOISE, updatemode='drag', marks={config.NOISE_MIN:{'label' : str(config.NOISE_MIN), 'style': {'color': '#f50'}},config.NOISE_MAX:{'label': str(config.NOISE_MAX), 'style': {'color': '#f50'}}})
                                        ])
                                ]),
                                #--- Display Selected values
                                html.Div(
                                    style={'display': 'none'},
                                    children=[
                                        html.P('Selected Data'),
                                        #html.Pre(id='selected-data')
                                        dash_table.DataTable(id='selected-data',
                                                             fixed_rows={ 'headers': True, 'data': 0 },
                                                             style_table={
                                                                     'height': '200px',
                                                                     #'overflowX': 'scroll',
                                                                     },
                                                             style_cell={
                                                                 'minWidth': '20px', 'width': '20px', 'maxWidth': '50px',
                                                                 'overflow': 'hidden',
                                                                 'textOverflow': 'ellipsis',
                                                                 })
                                        ]),
                                #--- Dropdown
                                html.Div(children=[
                                    html.Label('Select Inverse Function '),
                                    dcc.Dropdown(
                                            id='model-select',
                                            options=[
                                                    {'label':'--select--','value':'null'},
                                                    {'label': 'AE-MDN', 'value': 'AEMDN_n'}
                                                    ],
                                            value='null'
                                            )]),
                                #--- Submit & Reset button
                                html.Div(children=[
                                    #--- Submit Button
                                    html.Button(id='submit-button', className='cnf-button', n_clicks=0, children='Submit'),
                                    #--- Reset Button
                                    html.Button(id='reset-button', className='cnf-button', n_clicks=0, children='Reset'),
                                    ])
                                ])],
                        width=6,
                        ),
                #--- Output Graph
                dbc.Col(
                        html.Div(children=[
                                #--- Label
                                html.H5('PDF', className='h5-cnf'),
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
                                             ]),
                                html.Div([  
                                    html.Div(children=[
                                        html.Div([
                                            html.Label(['X',html.Sub('min')]),
                                            dcc.Input(id='Xmin_value', type='text', value=str(config.xMin), style={'margin-left':'8px','margin-right':'8px'}),
                                            html.Label(['X',html.Sub('max')]),
                                            dcc.Input(id='Xmax_value', type='text', value=str(config.xMax), style={'margin-left':'8px','margin-right':'8px'}),
                                            ],
                                            style={'display':'inline-block','padding-right':'10px','padding-bottom':'10px'},
                                            ),
                                        html.Div([
                                            html.Label(['Y',html.Sub('min')]),
                                            dcc.Input(id='Ymin_value', type='text', value=str(config.yMin), style={'margin-left':'8px','margin-right':'8px'}),
                                            html.Label(['Y',html.Sub('max')]),
                                            dcc.Input(id='Ymax_value', type='text', value=str(config.yMax), style={'margin-left':'8px','margin-right':'8px'}),
                                            ],
                                            style={'display': 'inline-block','padding-right':'10px','padding-bottom':'10px'},
                                            ),
                                        ]),
                                    html.Div(children=[
                                       dcc.RadioItems(
                                                id = 'scale-radio-buttons',
                                                options=[
                                                    {'label': 'Linear', 'value': 'lin'},
                                                    {'label': 'Log', 'value': 'log'}
                                                ],
                                                value='lin',
                                                labelStyle={'display': 'inline-block','padding-right':'10px'}
                                            ) 
                                       ]),
                                    html.Div(children=[
                                       html.Label('Select Graph Plot: '),
                                       dcc.Dropdown(
                                                id='op-plot-select',
                                                options=[
                                                        {'label': 'X vs Xu', 'value': 'PUA'},
                                                        {'label': 'X vs Xd', 'value': 'PDA'},
                                                        {'label': 'X vs Xu Ratio', 'value': 'PUR'},
                                                        {'label': 'X vs Xd Ratio', 'value': 'PDR'}
                                                        ],
                                                value='PUA'
                                                )
                                           ]),
                                    ]),
                                ]),
                        width=6,
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

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


#--- Toggle Modal Callback
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks"),Input('update-config','n_clicks')],
    [State("updateSample", "value"),
     State("updateNoiseMin", "value"),
     State("updateNoiseMax", "value"),
     State("modal", "is_open")],
)
def toggle_modal(n1, n2, n3, updateSampleValue, updateNoiseMin, updateNoiseMax, is_open):
    if n3:
        print('--> (RUNLOG) - updating Config....')
        updateConfig(updateSampleValue, updateNoiseMin, updateNoiseMax)
        updateUIcomponents()
        return not is_open
    if n1 or n2:
        return not is_open
    return is_open

#--- Update Config Submit
def updateConfig(updateSampleValue, updateNoiseMin, updateNoiseMax):
    config.setNOISE_MIN(int(updateNoiseMin))
    config.setNOISE_MAX(int(updateNoiseMax))
    config.setDEFAULT_SAMPLE(int(updateSampleValue))
    print('--> Updated Noise Min : ',config.NOISE_MIN)
    print('--> Updated Noise Max : ',config.NOISE_MAX)
    print('--> Updated Noise Sample : ',config.DEFAULT_SAMPLE)
    print('--> (RUNLOG) - config Updated.')
    return True
    #return not is_open

#--- Update Config Submit
def updateUIcomponents():
    return True

#--- display selected points
@app.callback(
    [Output('selected-data', 'data'),
     Output('selected-data', 'columns')
     ],
    [Input('g1', 'selectedData'),
     Input('ucert_value','value'),
     Input('f-value-slider','value'),
     Input('reset-button', 'n_clicks'),
     Input('xsec-data','value')])
def update_selected_data(selectedData, uncertainValue, noiseValue, reset_clicks, selectSample):
    #-- check if reset is clicked
    #--- validating the context to check which button is clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    #---if nothing is pressed display an empty Graph
    if button_id == 'No clicks yet':
        return [{}],[]
    
    #--- reset the graph if reset is clicked
    if button_id == 'reset-button':
        return [{}],[]
    #--- parsing the selected data
    else :
        #--- modify the data based on selected sample
        print('--> (RUNLOG) - sample to be selected - ',selectSample)
        global data, trueBar, xsecData
        trueBar = np.load(config.parPath)
        obs_p, obs_n = scaleCross(xsecData, selectSample)
        data['obs_p'] = obs_p #--- currently taking only one sample
        data['obs_n'] = obs_n #--- currently taking only one sample 
        #data['obs_p'] = obs_p[selectSample,:] #--- currently taking only one sample
        #data['obs_n'] = obs_n[selectSample,:] #--- currently taking only one sample 
        trueBar = trueBar[selectSample]
        s_data = json.dumps(selectedData, indent=2)
        if s_data != 'null':
            #--- set the data_Selected falg to true
            global dataSelected
            dataSelected = True
            selected = json.loads(s_data)
            #t_x = t_q2 = t_op = t_on = t_e_op = t_e_on = t_e_op_mod = t_e_on_mod = [] #--- create empty lists
            #--- add uncertainity
            data['err_obs_p'] = float(uncertainValue) * data['obs_p']
            data['err_obs_n'] = float(uncertainValue) * data['obs_n']
            data['err_obs_p_mod'] = data['err_obs_p'].copy()
            data['err_obs_n_mod'] = data['err_obs_n'].copy()
            #print('---> (RUNLOG) -----')
            #print(' | ','X',' | ','Q2',' | ','obs_p',' | ','obs_n',' | ','err_obs_p',' | ','err_obs_n',' | ','err_obs_p_mod',' | ','err_obs_n_mod',' | ')
            for point in selected['points']:
                #return point
                #--- update the is_selected column
                data.loc[point['pointIndex'],'is_selected'] = True
                #--- add noise to the points
                data.loc[point['pointIndex'],'err_obs_p_mod'] *= float(noiseValue)
                data.loc[point['pointIndex'],'err_obs_n_mod'] *= float(noiseValue)
                
                #print(' | ',point['pointIndex'],' | ',point['x'],' | ',point['y'],' | ', data.loc[point['pointIndex'], 'obs_p'],
                #  ' | ', data.loc[point['pointIndex'], 'obs_n'], ' | ', data.loc[point['pointIndex'], 'err_obs_p'], 
                #  ' | ', data.loc[point['pointIndex'], 'err_obs_n'], ' | ', data.loc[point['pointIndex'], 'err_obs_p_mod'],
                #  ' | ', data.loc[point['pointIndex'], 'err_obs_n_mod'], ' | ')
            print('--> (RUNLOG) - Total ',len(data[data['is_selected'] == True]),' samples selected...')
            selectedDataDF = data[data['is_selected'] == True]
            selectedDataDF = selectedDataDF[['X','Q2','obs_p','obs_n','err_obs_p','err_obs_n','err_obs_p_mod','err_obs_n_mod']]
            return selectedDataDF.to_dict('records'),[{"name":i,"id":i} for i in selectedDataDF.columns]
    return [{}],[]

#--- slider callback
@app.callback(
    Output('f-value-slider', 'value'),
    [Input('my-slider', 'value')])
def update_slider(value):
    return format(value)

#--- callback for Reset to clear dropdown
@app.callback(
            Output('model-select', 'value'),
        [Input('reset-button', 'n_clicks')])
def reset_data_dropdown(n_clicks):
    #--- make sure the data['is_selected'] is returned to false
    data['is_selected'] = False
    data['err_obs_p'] = 0
    data['err_obs_n'] = 0
    data['err_obs_p_mod'] = 0
    data['err_obs_n_mod'] = 0
    return ""

#--- callback for Reset to refresh the graph


#--- callback after submit
@app.callback(
        Output('output-graph-g2', 'figure'),
        [Input('submit-button', 'n_clicks'), #--- submit-button
         Input('reset-button', 'n_clicks'),  #--- Reset Button
         Input('Xmin_value', 'value'),Input('Xmax_value', 'value'), #--- Xmin,Xmax
         Input('Ymin_value', 'value'),Input('Ymax_value', 'value'), #--- Ymin,Ymax
         Input('scale-radio-buttons', 'value'), #--- Scale Radio
         Input('op-plot-select', 'value'), #--- Plot Select
         ],
        [
         State('selected-data', 'data'),
         State('ucert_value','value'),
         State('model-select','value'),
         State('f-value-slider', 'value')])
def update_outputGraph(submit_clicks, reset_clicks, xMin, xMax, yMin, yMax, scale_value, plotDisplay, tabledata, uncertainity_value, model_select, f_value):
    #--- validating the context to check which button is clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    #---if nothing is pressed display an empty Graph
    if button_id == 'No clicks yet':
        return go.Figure(
             data=[],
             layout=go.Layout(
                 title = 'No Input From User',
                 xaxis_title="X",
                 )
             )
    
    #--- reset the graph if reset is clicked
    if button_id == 'reset-button':
        return go.Figure(
             data=[],
             layout=go.Layout(
                 title = 'Graph Data Reset',
                 xaxis_title="X",
                 )
             )
    
    #if button_id == 'submit-button':
    # --- update the values
    if tabledata is not None:
        if model_select != 'null':
            #data_new = pd.DataFrame(tabledata)
            #--- udpate the dataframe and display all the columns
            
            #-- file name to be saved
            fname = 'test_backward'
            
            #-- check if No data is selected apply noise to every column
            global dataSelected
            if not dataSelected :
                data['is_selected'] = True
                dataSelected = True
                print('--> (RUNLOG) - No Data is selected, so applying noise to all columns.')
            #--- concatenate the lists to get the cross-sectional Data
            xsec = addNoise(data)
            #--- Predic the Noise From Data
            nn_pred = backwardPredict(fname, model_select, xsec)
            #nn_pred = unscaleOutput(nn_pred)
            
            #--- use the predicted values to get the pdf-up and pdf-down values
            pdf_pred_dict = calculate_pdf(nn_pred)
            pdf_true_dict = calculate_pdf(trueBar, pred=False)
            
            #--- get all the plot related Data
            yTrue, upperBound, yPred, lowerBound, plotTitle = getPlotData(plotDisplay, pdf_true_dict, pdf_pred_dict)
            
            #--- generate figure
            outputGraph = generatePDFplots(pdf_true_dict['x-axis'], yTrue, upperBound, yPred, lowerBound, plotTitle, scale_value)
            if plotDisplay == 'PUR' or plotDisplay == 'PDR' or plotDisplay == 'PUA' or plotDisplay == 'PDA' :
                outputGraph.update_xaxes(range=[float(xMin),float(xMax)])
                outputGraph.update_yaxes(range=[float(yMin),float(yMax)])
            return outputGraph
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


if __name__ == '__main__':
    app.run_server(debug = False)
    
'''
if (xMin != '' or xMax != '' or yMin != '' or yMax != ''):
                    print('--')
                    #--- do nothing
                else:
'''