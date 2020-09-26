# Dash
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table as dt
import dash
# Plotly
import plotly.express as px
import plotly.graph_objects as go
# Pandas
import pandas as pd
pd.options.plotting.backend = "plotly"
# Manejo de fechas
import datetime
# Files System Paths
from os import path, listdir
import re
from pathlib import Path
from glob import glob

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.GRID]

folder_name = "input/stations/"
file_name = "cat.csv"
file_path = path.join(path.dirname(__file__), folder_name, file_name)
stations = pd.read_csv(file_path)
stations = stations["station"].unique()
stations = [{"label":s, "value":s} for s in stations]

app = DjangoDash(__name__.split(".")[-1], serve_locally=True, add_bootstrap_links=True)

filter_stations = dbc.FormGroup(
    [
        dbc.Label("Estaciones: "),
        dcc.Dropdown(
            id='stations',
            options=stations,
            # multi=True,
            value=None,
            placeholder="Seleccione una estación",
        ),
    ]
)

filter_button = dbc.Button(
    "Filtrar",
    id="filter_button",
    color="primary",
)


html_filters = dbc.Card(
    [
        filter_stations,
        filter_button,
    ],
    body=True,
    # style={"position":"sticky", "top":0},
    className="w-75 mb-3"
)

html_table = dt.DataTable(
    id='overview-datatable',
    sort_action="native",
    # style_as_list_view=True,
    style_table={'maxWidth': '95%'},
    style_data={'whiteSpace': 'normal',
                'height': 'auto',},
    style_header={'backgroundColor': 'rgb(200, 212, 227)',
                  'color': 'rgb(42, 63, 95)',
                  'fontWeight': 'bold'},
    style_cell={'backgroundColor': 'rgb(235, 240, 248)',
                'color': 'rgb(42, 63, 95)',
                'fontSize': 12,
                'textAlign': 'left',
                'width': '50px',
                'border': '1px solid grey'},
)


overview = dbc.Card(
                [
                    dbc.CardHeader(
                        html.H2(
                            dbc.Button(
                                "Vista rápida",
                                color="link",
                                id="overview-toggle",
                            )
                        )
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            html.Div(
                                children=[
                                    html.Div(
                                        [
                                            dcc.Loading(
                                                id="overview-loading1",
                                                children=
                                                    [
                                                        html.Div([html_table], id='overview-table'),
                                                        # html.Div(id='graph'),
                                                    ],
                                                    type="default",
                                                    # fullscreen=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        [
                                            dcc.Loading(
                                                id="overview-loading2",
                                                children=
                                                    [
                                                        # html.Div([html_table], id='table'),
                                                        html.Div(id='overview-graph'),
                                                    ],
                                                    type="graph",
                                                    # fullscreen=False,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        id="collapse-overview",
                    ),
                ]
            )

features_selection = dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H2(
                                    dbc.Button(
                                        "Selección de características",
                                        color="link",
                                        id="features-selection-toggle",
                                    )
                                )
                            ),
                            dbc.Collapse(
                                dbc.CardBody(
                                    html.Div(
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div("One of two columns"), width=4),
                                                dbc.Col(html.Div("One of two columns"), width=4),
                                            ],
                                            justify="center",
                                        )
                                    ),
                                ),
                                id="collapse-features-selection",
                            ),
                        ]
                    )

model = dbc.Card(
            [
                dbc.CardHeader(
                    html.H2(
                        dbc.Button(
                            "Modelado",
                            color="link",
                            id="model-toggle",
                        )
                    )
                ),
                dbc.Collapse(
                    dbc.CardBody(
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(html.Div("One of two columns"), width=4),
                                    dbc.Col(html.Div("One of two columns"), width=4),
                                ],
                                justify="center",
                            )
                        ),
                    ),
                    id="collapse-model",
                ),
            ]
        )

validate = dbc.Card(
            [
                dbc.CardHeader(
                    html.H2(
                        dbc.Button(
                            "Validación",
                            color="link",
                            id="validate-toggle",
                        )
                    )
                ),
                dbc.Collapse(
                    dbc.CardBody(
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(html.Div("One of two columns"), width=4),
                                    dbc.Col(html.Div("One of two columns"), width=4),
                                ],
                                justify="center",
                            )
                        ),
                    ),
                    id="collapse-validate",
                ),
            ]
        )

accordion = html.Div([overview, features_selection, model, validate], className="accordion")

# html_body = html.Div(
#     children=[
#         html.Div(
#             [
#                 dcc.Loading(
#                     id="overview-loading1",
#                     children=
#                     [
#                         html.Div([html_table], id='table'),
#                         # html.Div(id='graph'),
#                     ],
#                     type="default",
#                     # fullscreen=False,
#                 ),
#             ],
#         ),
#         html.Div(
#             [
#                 dcc.Loading(
#                     id="overview-loading2",
#                     children=
#                     [
#                         # html.Div([html_table], id='table'),
#                         html.Div(id='graph'),
#                     ],
#                     type="graph",
#                     # fullscreen=False,
#                 ),
#             ],
#         ),
#     ],
# ),

app.layout = dbc.Container(
    [
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html_filters, md=5)
            ],
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(accordion, md=10)
            ],
            justify="center",
        ),
    ],
    fluid=True,
)

def read_data(selected_station):
    folder_name = "input/stations/hourly/"
    file_name = f"data_{selected_station}.csv"
    data_path = path.join(path.dirname(__file__), folder_name, file_name)

    df = pd.read_csv(data_path, header=[0,1,2], index_col=0, parse_dates=True)
    df.columns = df.columns.droplevel(level=[0,2])

    return df

def data_table(df):
    df_table = pd.concat([df.apply(pd.Series.first_valid_index).to_frame(name="Primer Dato"),
                          df.apply(pd.Series.last_valid_index).to_frame(name="Ultimo Dato")], axis=1, sort=False)

    df_table['Espacio muestral'] = (df_table["Ultimo Dato"] - df_table["Primer Dato"]).astype('timedelta64[s]')//3600.0

    df_table['Muestras reales'] = [df_table.iloc[v, 2] - df.loc[df_table.iloc[v, 0]: df_table.iloc[v, 1], df_table.index[v]].isna().sum() for v in range(len(df_table.index))]

    df_table.index = df_table.index.set_names(['Variable'])
    df_table = df_table.reset_index(level=[0])
    df_table['id'] = df_table.index

    return df_table

def sample_space_bars(df, column):
    n_bins = 10
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    col_max = df[column].max()
    col_min = 0 #df[column].min()
    ranges = [
        ((col_max - col_min) * i) + col_min
        for i in bounds
    ]
    # Al menos el 70% del rango muestral maximo
    require_samples = col_max * 0.7
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound} && {{{column}}} < {require_samples}')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound, require_samples=require_samples),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #FF9B9B 0%,
                    #FF9B9B {max_bound_percentage}%,
                    #FFC8C8 {max_bound_percentage}%,
                    #FFC8C8 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound} && {{{column}}} >= {require_samples}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound, require_samples=require_samples),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #9B9BFF 0%,
                    #9B9BFF {max_bound_percentage}%,
                    #C8C8FF {max_bound_percentage}%,
                    #C8C8FF 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles

def real_sample_bars(df, column):
    styles = []
    for r in range(len(df)):
        n_bins = 10
        bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
        if pd.isna(df.loc[r, 'Espacio muestral']):
            col_max = 0
        else:
            col_max = float(df.loc[r, 'Espacio muestral'])
        col_min = 0 #df[column].min()
        ranges = [
            ((col_max - col_min) * i) + col_min
            for i in bounds
        ]
        # Al menos el 70% del rango muestral maximo de cada variable
        require_samples = col_max * 0.7
        value = float(df.loc[r, 'Muestras reales'])
        # print(col_max, require_samples, value)
        # styles = []

        for i in range(1, len(bounds)):
            min_bound = ranges[i - 1]
            max_bound = ranges[i]
            max_bound_percentage = bounds[i] * 100

            styles.append({
                'if': {
                    'filter_query': (
                        '{{id}} = {r}' +
                        ' && {value} >= {min_bound}' +
                        (' && {value} < {max_bound} && {value} < {require_samples}')
                    ).format(r=r, value=value, min_bound=min_bound, max_bound=max_bound, require_samples=require_samples),
                    'column_id': column
                },
                'background': (
                    """
                        linear-gradient(90deg,
                        #FF9B9B 0%,
                        #FF9B9B {max_bound_percentage}%,
                        #FFC8C8 {max_bound_percentage}%,
                        #FFC8C8 100%)
                    """.format(max_bound_percentage=max_bound_percentage)
                ),
                'paddingBottom': 2,
                'paddingTop': 2
            })

            styles.append({
                'if': {
                    'filter_query': (
                        '{{id}} = {r}' +
                        ' && {value} >= {min_bound}' +
                        (' && {value} < {max_bound} && {value} >= {require_samples}' if (i < len(bounds) - 1) else '')
                    ).format(r=r, value=value, min_bound=min_bound, max_bound=max_bound, require_samples=require_samples),
                    'column_id': column
                },
                'background': (
                    """
                        linear-gradient(90deg,
                        #9B9BFF 0%,
                        #9B9BFF {max_bound_percentage}%,
                        #C8C8FF {max_bound_percentage}%,
                        #C8C8FF 100%)
                    """.format(max_bound_percentage=max_bound_percentage)
                ),
                'paddingBottom': 2,
                'paddingTop': 2
            })
    return styles

@app.callback(
    [Output('overview-datatable', 'columns'),
     Output('overview-datatable', 'data'),
     Output('overview-datatable', 'style_data_conditional')],
    [Input('filter_button', 'n_clicks')],
    [State('stations', 'value')])

def update(n_clicks, selected_station):
    if selected_station is None:
        raise dash.exceptions.PreventUpdate

    df = read_data(selected_station)

    df_table = data_table(df)

    # "deletable": True
    columns = [{'id': c, 'name': c} for c in df_table.columns if c != 'id']

    data = df_table.to_dict('records')

    style_data_conditional = []

    sample_space_style = (sample_space_bars(df_table, 'Espacio muestral'))
    [style_data_conditional.append(style) for style in sample_space_style]
    

    real_sample_style = (real_sample_bars(df_table, 'Muestras reales'))
    [style_data_conditional.append(style) for style in real_sample_style]

    style_data_conditional.append({'if': {'state': 'active'},
                                   'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                   'border': '1px solid rgb(0, 116, 217)'})

    print('TABLA')

    return [columns, data, style_data_conditional]

@app.callback(
    Output('overview-graph', 'children'),
    [
        Input('filter_button', 'n_clicks')
    ],
    [
        State('stations', 'value')
    ],
)

def update_figure(n_clicks, selected_station):
    if selected_station is None:
        raise dash.exceptions.PreventUpdate
    
    df = read_data(selected_station)

    df_table = data_table(df)

    max_range = df_table['Espacio muestral'].max()
    data = []
    ranges = []

    for i in range(len(df_table)):
        variable = df_table.iloc[i, 0]
        initial_date = df_table.iloc[i, 1]
        finish_date = df_table.iloc[i, 2]
        sample_range = df_table.iloc[i, 3]
        real_sample = df_table.iloc[i, 4]
        range_pct = (sample_range * 100 / max_range)
        sample_pct = (real_sample * 100 / sample_range)
        ranges.append(range_pct)
        if not pd.isna(real_sample):
            finish_sample_date = initial_date + datetime.timedelta(days=real_sample / 24.0)
        else:
            finish_sample_date = initial_date
        data.append(dict(Variable=variable, Inicio=initial_date, Fin=finish_date, Fin_Muestra=finish_sample_date, Pct_Muestral=range_pct, Pct_Real=sample_pct),)

    df_graph = pd.DataFrame(data)

    print(df_graph)

    max_color = "#C8C8FF"
    min_color = "#FFC8C8"

    scale = [(0, min_color), (0.69, min_color), (0.7, max_color), (1, max_color)]

    time_line = px.timeline(df_graph, x_start="Inicio", x_end="Fin", y="Variable", color="Pct_Muestral",
                      color_continuous_scale=scale, range_color=[0, 100],
                      labels={"Variable": "Variable", "Pct_Muestral": "%"})

    time_line.update_yaxes(autorange="reversed")

    time_line.update_layout(title_text="Espacio muestral total por variable",
                      title_x=0.5,
                      title_y=0.92)

    for i in range(len(df_graph)):
        variable = df_graph.iloc[i, 0]
        date_ini = df_graph.iloc[i, 1]
        date_end = df_graph.iloc[i, 3]
        sample_pct = float(df_graph.iloc[i, 5])

        if sample_pct >= 70.0:
            color='#9B9BFF'
        else:
            color='#FF9B9B'

        time_line.add_trace(
            go.Scatter(
                x=[date_ini, date_end],
                y=[variable, variable],
                mode="lines",
                line=dict(color=color, width=7),
                hovertext=[f"{sample_pct:.0f}%", f"{sample_pct:.0f}%"],
                hoverinfo="text",
                showlegend=False)
        )

    cor = px.imshow(df.corr(), color_continuous_scale ='RdBu')

    cor.update_layout(title={'text': "Correlación",
                             'y':0,
                             'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})
    
    cor.update_xaxes(side="top")

    graphs = html.Div([html.Div(dcc.Graph(figure=time_line)),
                       html.Div(dcc.Graph(figure=cor))])

    print('GRAFICA')

    return graphs


@app.callback(
    [Output("collapse-overview", "is_open"),
     Output("collapse-features-selection", "is_open"),
     Output("collapse-model", "is_open"),
     Output("collapse-validate", "is_open")],
    [Input('filter_button', 'n_clicks'),
     Input("overview-toggle", "n_clicks"),
     Input("features-selection-toggle", "n_clicks"),
     Input("model-toggle", "n_clicks"),
     Input("validate-toggle", "n_clicks")],
    [State("collapse-overview", "is_open"),
     State("collapse-features-selection", "is_open"),
     State("collapse-model", "is_open"),
     State("collapse-validate", "is_open")],
)
def toggle_accordion(filter_click,
                     overview_click, features_selection_click, model_click, validate_click,
                     is_open_overview, is_open_features_selection, is_open_model, is_open_validate):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "overview-toggle" and overview_click:
        is_open_overview = not is_open_overview
    elif button_id == "features-selection-toggle" and features_selection_click:
        is_open_features_selection = not is_open_features_selection
    elif button_id == "model-toggle" and model_click:
        is_open_model = not is_open_model
    elif button_id == "validate-toggle" and validate_click:
        is_open_validate = not is_open_validate
    elif button_id == "filter_button":
        is_open_overview = True
        is_open_features_selection = True
        is_open_model = True
        is_open_validate = True
    return is_open_overview, is_open_features_selection, is_open_model, is_open_validate