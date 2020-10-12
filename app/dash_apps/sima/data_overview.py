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
# Numpy
import numpy as np
# Manejo de fechas
import datetime
# Files System Paths
from os import path, listdir
import re
from pathlib import Path
from glob import glob
# Libreria del modelo
import xgboost
# Metricas
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.metrics import r2_score, explained_variance_score
from scipy.stats import pearsonr
# Procesamiento
import multiprocessing

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.GRID]

folder_name = "input/stations/"
file_name = "cat.csv"
file_path = path.join(path.dirname(__file__), folder_name, file_name)
stations = pd.read_csv(file_path)
stations = stations["station"].unique()
stations = [{"label":s, "value":s} for s in stations]

df = None
df_table = None
df_data_model = pd.DataFrame()
df_data_target = pd.DataFrame()

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
        dbc.Label("Frecuencia de muestreo"),
        dbc.RadioItems(
            options=[
                {"label": "Horaria", "value": 0},
                {"label": "Diaria", "value": 1}
            ],
            value=0,
            id="frequency-radioitems",
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

overview_table = dt.DataTable(
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
                                            html.Div([overview_table]),
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
                                            # html.Div([overview_table], id='table'),
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

features_table = dt.DataTable(
    id='features-datatable',
    sort_action="native",
    # style_as_list_view=True,
    style_table={'maxWidth': '50%'},
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
    row_selectable="multi",
)

target_table = dt.DataTable(
    id='target-datatable',
    sort_action="native",
    # style_as_list_view=True,
    style_table={'maxWidth': '50%'},
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
    row_selectable="single",
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
                            dbc.Col(
                                html.Div(
                                    [
                                        html.P("Selecciona las características"),
                                        features_table
                                    ]
                                ),
                                width=5
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.P("Variable Objetivo PM10"),
                                        # target_table
                                    ]
                                ),
                                width=5
                            ),
                        ],
                        justify="center",
                    )
                ),
            ),
            id="collapse-features-selection",
        ),
    ]
)

impute_dataset = dbc.FormGroup(
    [
        # dbc.Label("Choose one"),
        dbc.RadioItems(
            options=[
                {"label": "Características", "value": 0},
                {"label": "Objetivo", "value": 1},
            ],
            value=0,
            id="impute-dataset-radioitems",
        ),
    ]
)

impute_data = dbc.FormGroup(
    [
        # dbc.Label("Choose one"),
        dbc.RadioItems(
            options=[
                {"label": "Promedio Vecinos mas Cercanos", "value": 0},
                {"label": "Interpolación Lineal", "value": 1},
                {"label": "Interpolación Cuadrática", "value": 2},
                {"label": "Interpolación Cúbica", "value": 3},
                {"label": "Promedio por Periodo", "value": 4},
                {"label": "Eliminar (Se elimina el mismo perido en las características y el objetivo)", "value": 5}
            ],
            value=4,
            id="impute-radioitems",
        ),
        dbc.InputGroup(
            [
                html.P("Ingrese la cantidad de vecinos a considerar", id="impute-label"),
                dbc.Input(id="impute-parameter", type="number", min=1),
                # dbc.InputGroupAddon("$", addon_type="prepend"),
                # dbc.Input(placeholder="Amount", type="number"),
                # dbc.InputGroupAddon(".00", addon_type="append"),
            ],
            className="mb-3",
        ),
        # html.P(html.Br()),
        dbc.Button(
            "Reiniciar",
            id="impute_reset_button",
            color="danger",
            className="mr-1"
        ),
        dbc.Button(
            "Aplicar",
            id="impute_apply_button",
            color="primary",
            className="mr-1"
        ),
    ]
)

impute_table = dt.DataTable(
    id='impute-datatable',
    sort_action="native",
    # style_as_list_view=True,
    style_table={'maxWidth': '50%'},
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

impute = dbc.Card(
    [
        dbc.CardHeader(
            html.H2(
                dbc.Button(
                    "Valores faltantes",
                    color="link",
                    id="impute-toggle",
                )
            )
        ),

        dbc.Collapse(
            dbc.CardBody(
                html.Div(
                    children=[
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.P("Selecciona el conjunto de datos", id="impute-dataset-label"),
                                                impute_dataset,
                                                html.P("Selecciona el método de imputación de datos"),
                                                impute_data
                                            ]
                                        ),
                                        width=8
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dcc.Loading(
                                                    id="impute-table-loading",
                                                    children=
                                                        [
                                                            html.Div(
                                                                [
                                                                    impute_table
                                                                ]
                                                            ),
                                                        ],
                                                    type="default",
                                                    # fullscreen=False,
                                                ),
                                            ],
                                        ),
                                        width=4
                                    ),
                                ],
                                justify="center",
                            )
                        ),
                        html.Div(
                            [
                                dcc.Loading(
                                    id="impute-graph-loading",
                                    children=
                                        [
                                            # html.Div([overview_table], id='table'),
                                            html.Div(id='impute-graph'),
                                        ],
                                    type="graph",
                                    # fullscreen=False,
                                ),
                            ],
                        ),
                    ],
                ),
            ),
            id="collapse-impute",
        )
    ]
)

model_data = dbc.FormGroup(
    [
        dbc.InputGroup(
            [
                html.P(["Ingrese hasta que fecha (exclusive) se utilizará para entrenamiento", html.Br(), "(el resto se utilizará para validación)"], id="model-split-label"),
                dcc.DatePickerSingle(
                    id='model-date-split',
                    min_date_allowed=datetime.datetime(2015, 1, 1),
                    max_date_allowed=datetime.datetime(2020, 12, 1),
                    initial_visible_month=datetime.datetime(2019, 12, 31),
                    date=str(datetime.datetime(2020, 1, 1))
                )
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                html.P(["Profundidad máxima de los árboles de decision"], id="model-max-depth-label"),
                dbc.Input(id="model-min-depth-parameter", type="number", min=0, value=9),
                dbc.Input(id="model-max-depth-parameter", type="number", min=0, value=12),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                html.P(["Peso mínimo de los nodos hijos"], id="model-min-child-weight-label"),
                dbc.Input(id="model-min-child-weight-parameter", type="number", min=0, value=5, step='any'),
                dbc.Input(id="model-max-child-weight-parameter", type="number", min=0, value=8, step='any'),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                html.P(["Tasa de aprendizaje"], id="model-eta-label"),
                dbc.Input(id="model-min-eta-parameter", type="number", min=0, max=1, value=0.3, step='any'),
                dbc.Input(id="model-max-eta-parameter", type="number", min=0, max=1, value=0.3, step='any'),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                html.P(["Tamaño de submuestras de entrenamiento"], id="model-subsample-label"),
                dbc.Input(id="model-min-subsample-parameter", type="number", min=0, max=1, value=0.7, step='any'),
                dbc.Input(id="model-max-subsample-parameter", type="number", min=0, max=1, value=1, step='any'),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                html.P(["Tasa de columnas aleatorias para entrenamiento"], id="model-colsample-bytree-label"),
                dbc.Input(id="model-min-colsample-bytree-parameter", type="number", min=0, max=1, value=0.7, step='any'),
                dbc.Input(id="model-max-colsample-bytree-parameter", type="number", min=0, max=1, value=1, step='any'),
            ],
            className="mb-3",
        ),
        html.P(["Seleccione el método de evaluación"], id="model-eval-metric-label"),
        dbc.InputGroup(
            [
                dbc.RadioItems(
                    options=[
                        {"label": "RMSE", "value": 0},
                        {"label": "RMSLE", "value": 1},
                        {"label": "MAE", "value": 2},
                    ],
                    value=2,
                    id="model-eval-metric-radioitems",
                ),
            ],
            className="mb-3",
        ),
        # html.P(html.Br()),
        dbc.Button(
            "Aplicar",
            id="model_apply_button",
            color="primary",
            className="mr-1"
        ),
    ]
)

model_table = dt.DataTable(
    id='model-datatable',
    sort_action="native",
    # style_as_list_view=True,
    style_table={'maxWidth': '50%'},
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
                    children=[
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                model_data
                                            ]
                                        ),
                                        width=8
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dcc.Loading(
                                                    id="model-table-loading",
                                                    children=
                                                        [
                                                            html.Div(
                                                                [
                                                                    model_table
                                                                ]
                                                            ),
                                                        ],
                                                    type="default",
                                                    # fullscreen=False,
                                                ),
                                            ],
                                        ),
                                        width=4
                                    ),
                                ],
                                justify="center",
                            )
                        ),
                    ],
                )
            ),
            id="collapse-model",
        ),
    ]
)

validate_table = dt.DataTable(
    id='validate-datatable',
    sort_action="native",
    # style_as_list_view=True,
    style_table={'maxWidth': '50%'},
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
            html.Div(
                    children=[
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.P("Métricas"),
                                                dcc.Loading(
                                                    id="validate-table-loading",
                                                    children=
                                                        [
                                                            html.Div(
                                                                [
                                                                    validate_table
                                                                ]
                                                            ),
                                                        ],
                                                    type="default",
                                                    # fullscreen=False,
                                                ),
                                            ],
                                        ),
                                        width=5
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                # dcc.Loading(
                                                #     id="validate-table-loading",
                                                #     children=
                                                #         [
                                                #             html.Div(
                                                #                 [
                                                #                     validate_table
                                                #                 ]
                                                #             ),
                                                #         ],
                                                #     type="default",
                                                #     # fullscreen=False,
                                                # ),
                                            ],
                                        ),
                                        width=5
                                    ),
                                ],
                                justify="center",
                            )
                        ),
                        html.Div(
                            [
                                dcc.Loading(
                                    id="validate-graph-loading",
                                    children=
                                        [
                                            # html.Div([overview_table], id='table'),
                                            html.Div(id='validate-graph'),
                                        ],
                                    type="graph",
                                    # fullscreen=False,
                                ),
                            ],
                        ),
                    ],
                ),
            id="collapse-validate",
        ),
    ]
)

accordion = html.Div([overview, features_selection, impute, model, validate], className="accordion")

# html_body = html.Div(
#     children=[
#         html.Div(
#             [
#                 dcc.Loading(
#                     id="overview-loading1",
#                     children=
#                     [
#                         html.Div([overview_table], id='table'),
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
#                         # html.Div([overview_table], id='table'),
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

def read_data(selected_station, frequency):
    if frequency == 0:
        freq_folder = "hourly"
        freq = "H"
    elif frequency == 1:
        freq_folder = "daily"
        freq = "D"

    folder_name = f"input/stations/{freq_folder}/"
    # file_name = f"data_{selected_station}_avg.csv"
    file_name = f"data_{selected_station}.csv"
    data_path = path.join(path.dirname(__file__), folder_name, file_name)

    df = pd.read_csv(data_path, header=[0,1,2], index_col=0, parse_dates=True)
    df.columns = df.columns.droplevel(level=[0,2])
    df = df.asfreq(freq)

    years = [2015, 2016, 2017, 2018, 2019, 2020]

    # Se descarta el 29 de febrero de los años bisiestos
    for year in years:
        try:
            ini_date = datetime.datetime(year, 2, 29)
            range_dates = pd.date_range(start=ini_date, freq='H', periods=24)
            df = df[~df.index.isin(range_dates)]
        except ValueError as e:
                pass

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

# @app.callback(
#     [Output('overview-datatable', 'columns'),
#      Output('overview-datatable', 'data'),
#      Output('overview-datatable', 'style_data_conditional')],
#     [Input('filter_button', 'n_clicks')],
#     [State('stations', 'value')])

# def update(n_clicks, selected_station):
#     if selected_station is None:
#         raise dash.exceptions.PreventUpdate

#     df = read_data(selected_station)

#     df_table = data_table(df)

#     # "deletable": True
#     columns = [{'id': c, 'name': c} for c in df_table.columns if c != 'id']

#     data = df_table.to_dict('records')

#     style_data_conditional = []

#     sample_space_style = (sample_space_bars(df_table, 'Espacio muestral'))
#     [style_data_conditional.append(style) for style in sample_space_style]
    

#     real_sample_style = (real_sample_bars(df_table, 'Muestras reales'))
#     [style_data_conditional.append(style) for style in real_sample_style]

#     style_data_conditional.append({'if': {'state': 'active'},
#                                    'backgroundColor': 'rgba(0, 116, 217, 0.3)',
#                                    'border': '1px solid rgb(0, 116, 217)'})

#     print('TABLA')

#     return [columns, data, style_data_conditional]

# @app.callback(
#     Output('overview-graph', 'children'),
#     [
#         Input('filter_button', 'n_clicks')
#     ],
#     [
#         State('stations', 'value')
#     ],
# )

# def update_figure(n_clicks, selected_station):
#     if selected_station is None:
#         raise dash.exceptions.PreventUpdate
    
#     df = read_data(selected_station)

#     df_table = data_table(df)

#     max_range = df_table['Espacio muestral'].max()
#     data = []
#     ranges = []

#     for i in range(len(df_table)):
#         variable = df_table.iloc[i, 0]
#         initial_date = df_table.iloc[i, 1]
#         finish_date = df_table.iloc[i, 2]
#         sample_range = df_table.iloc[i, 3]
#         real_sample = df_table.iloc[i, 4]
#         range_pct = (sample_range * 100 / max_range)
#         sample_pct = (real_sample * 100 / sample_range)
#         ranges.append(range_pct)
#         if not pd.isna(real_sample):
#             finish_sample_date = initial_date + datetime.timedelta(days=real_sample / 24.0)
#         else:
#             finish_sample_date = initial_date
#         data.append(dict(Variable=variable, Inicio=initial_date, Fin=finish_date, Fin_Muestra=finish_sample_date, Pct_Muestral=range_pct, Pct_Real=sample_pct),)

#     df_graph = pd.DataFrame(data)

#     print(df_graph)

#     max_color = "#C8C8FF"
#     min_color = "#FFC8C8"

#     scale = [(0, min_color), (0.69, min_color), (0.7, max_color), (1, max_color)]

#     time_line = px.timeline(df_graph, x_start="Inicio", x_end="Fin", y="Variable", color="Pct_Muestral",
#                       color_continuous_scale=scale, range_color=[0, 100],
#                       labels={"Variable": "Variable", "Pct_Muestral": "%"})

#     time_line.update_yaxes(autorange="reversed")

#     time_line.update_layout(title_text="Espacio muestral total por variable",
#                       title_x=0.5,
#                       title_y=0.92)

#     for i in range(len(df_graph)):
#         variable = df_graph.iloc[i, 0]
#         date_ini = df_graph.iloc[i, 1]
#         date_end = df_graph.iloc[i, 3]
#         sample_pct = float(df_graph.iloc[i, 5])

#         if sample_pct >= 70.0:
#             color='#9B9BFF'
#         else:
#             color='#FF9B9B'

#         time_line.add_trace(
#             go.Scatter(
#                 x=[date_ini, date_end],
#                 y=[variable, variable],
#                 mode="lines",
#                 line=dict(color=color, width=7),
#                 hovertext=[f"{sample_pct:.0f}%", f"{sample_pct:.0f}%"],
#                 hoverinfo="text",
#                 showlegend=False)
#         )

#     cor = px.imshow(df.corr(), color_continuous_scale ='RdBu')

#     cor.update_layout(title={'text': "Correlación",
#                              'y':0,
#                              'x':0.5,
#                              'xanchor': 'center',
#                              'yanchor': 'top'})
    
#     cor.update_xaxes(side="top")

#     graphs = html.Div([html.Div(dcc.Graph(figure=time_line)),
#                        html.Div(dcc.Graph(figure=cor))])

#     print('GRAFICA')

#     return graphs


def update_overview_table(df_table):
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

    # print('TABLA')

    return [columns, data, style_data_conditional]

def update_overview_figures(df, df_table):
    print("OVERVIEW FIGURES")
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

    # print(df_graph)

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

    # print('GRAFICA')

    return [graphs]

def update_features_table(df, selected_target=None):

    selected_target = "PM10"
    features = df.columns.difference([selected_target])

    correlations = {}
    for f in features:
        data_temp = df[[f, selected_target]].dropna()
        if data_temp.empty:
            correlations[f] = 0
        else:
            x1 = data_temp[f].values
            x2 = data_temp[selected_target].values
            correlations[f] = pearsonr(x1, x2)[0]

    data_correlations = pd.DataFrame(correlations, index=['Correlacion']).T
    data_correlations['Variable'] = data_correlations.index
    data_correlations.reset_index(drop=True, inplace=True)
    data_correlations = data_correlations[['Variable', 'Correlacion']]

    data = data_correlations.loc[data_correlations['Correlacion'].abs().sort_values(ascending=False).index].to_dict('records')
    # "deletable": True 
    # columns = [{'id': c, 'name': c} for c in df_table.columns if c != 'id']
    columns = [{'id': "Variable", 'name': "Variable"},
               {'id': "Correlacion", 'name': "Correlacion"}]

    # data = df_table[~df_table["Variable"].isin(['PM10'])].to_dict('records')

    # if selected_target is not None:
    #     data = df_table[["Variable"]].drop(df_table.index[selected_target]).to_dict('records')
    # else:
    #     data = df_table[["Variable"]].to_dict('records')
    
    # print(data)

    style_data_conditional = []

    style_data_conditional.append({'if': {'state': 'active'},
                                   'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                   'border': '1px solid rgb(0, 116, 217)'})

    # print("TABLA FEATURES")

    return [columns, data, style_data_conditional]

# def update_target_table(df_table):
#     # "deletable": True
#     # columns = [{'id': c, 'name': c} for c in df_table.columns if c != 'id']
#     columns = [{'id': "Variable", 'name': "Variable"}]

#     data = df_table[["Variable"]].to_dict('records')

#     # print(data)

#     style_data_conditional = []

#     style_data_conditional.append({'if': {'state': 'active'},
#                                    'backgroundColor': 'rgba(0, 116, 217, 0.3)',
#                                    'border': '1px solid rgb(0, 116, 217)'})

#     # print("TABLA TARGET")

#     return [columns, data, style_data_conditional]

def update_impute_table(df_data):
    # "deletable": True
    # columns = [{'id': c, 'name': c} for c in df_table.columns if c != 'id']
    columns = [{'id': "Variable", 'name': "Variable"},
               {'id': "NA", 'name': "NA"}]
    df_table = df_data.isna().sum()
    df_table = df_table.to_frame('NA')
    df_table.index = df_table.index.set_names(['Variable'])
    df_table = df_table.reset_index(level=[0])

    data = df_table.to_dict('records')

    style_data_conditional = []

    style_data_conditional.append({'if': {'state': 'active'},
                                   'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                   'border': '1px solid rgb(0, 116, 217)'})

    # print("TABLA TARGET")

    return [columns, data, style_data_conditional]

def update_impute_figures(df):
    fig = go.Figure()
    for column in df:
        trace = go.Scatter(x=df.index,
                           y=df[column],
                           mode="lines",
                           name=column)
        fig.add_trace(trace)

    # fig['layout']['xaxis']['type'] = 'category'

    graphs = html.Div([html.Div(dcc.Graph(figure=fig))])

    # print('GRAFICA')

    return [graphs]

# Opcion 1
def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n/2)
            lower = np.max([0, int(i-n_by_2)])
            upper = np.min([len(ts)+1, int(i+n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out

# Opcion 2-4
def interpolate(ts, method):
    out = np.copy(ts)
    out = ts.interpolate(method=method)
    return out

# Opcion 5
def periods_mean(ts, n=None):
    out = ts.copy()
    nan = out[out.isnull()]
    for nan_date in nan.index:
        ini_date = datetime.datetime(2015, nan_date.month, nan_date.day, nan_date.hour)
        range_dates = pd.date_range(start=ini_date, freq=pd.DateOffset(years=1), periods=6)
        data = []
        for df_date in range_dates:
            try:
                data.append(out.loc[df_date])
            except KeyError as e:
                pass
        value = np.nanmean(data)
        out[nan_date] = value
    return out

@app.callback(
    [# Overview
     # Data table
     Output('overview-datatable', 'columns'),
     Output('overview-datatable', 'data'),
     Output('overview-datatable', 'style_data_conditional'),
     # Graphs
     Output('overview-graph', 'children'),
     # Features Selection
     # Features table
     Output('features-datatable', 'columns'),
     Output('features-datatable', 'data'),
     Output('features-datatable', 'style_data_conditional'),
    #  Output('features-datatable', "selected_row_ids"),
     # Features Selection
     # Target table
    #  Output('target-datatable', 'columns'),
    #  Output('target-datatable', 'data'),
    #  Output('target-datatable', 'style_data_conditional'),
     # Impute
     # NA's table
     Output('impute-datatable', 'columns'),
     Output('impute-datatable', 'data'),
     Output('impute-datatable', 'style_data_conditional'),
     # Graphs
     Output('impute-graph', 'children'),
     ],
    [Input('filter_button', 'n_clicks'),
     Input('stations', 'value'),
     Input("frequency-radioitems",'value'),
     Input('features-datatable', "selected_rows"),
    # Dataset
     Input('impute-dataset-radioitems', 'value'),
     Input('impute_apply_button', 'n_clicks'),
     Input('impute_reset_button', 'n_clicks'),],
    [State('features-datatable', "derived_virtual_data"),
    #  State('features-datatable', "selected_rows"),
     State('impute-radioitems', "value"),
     State('impute-parameter', 'value')],
)

def process_data(filter_clicks, selected_station, selected_frequency,
                 selected_features_id, #select_target,
                 impute_dataset, impute_apply_clicks, impute_reset_clicks,
                 features_rows, #selected_features_id,
                 impute_selected, impute_parameter):

    if selected_station is None:
        raise dash.exceptions.PreventUpdate
        return

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    print(button_id, selected_features_id)

    result = [dash.no_update] * 11

    global df, df_table, df_data_model, df_data_target
        # print(selected_features)
        # print(features_rows)
    
    if button_id == "stations" or button_id == "frequency-radioitems":
        df = None
        df_table = None
        df_data_model = pd.DataFrame()
        df_data_target = pd.DataFrame()
    
        result = [''] * 11
    elif button_id == "filter_button" and filter_clicks:
        df = read_data(selected_station, selected_frequency)
        df_table = data_table(df)
        
        overview_table = update_overview_table(df_table)

        overview_figures = update_overview_figures(df, df_table)

        # features_table = update_features_table(df_table, selected_target)
        features_table = update_features_table(df)

        # target_table = update_target_table(df_table)
        df_data_target = df[["PM10"]]

        result = overview_table + overview_figures + features_table + [''] * 4
    elif button_id == "features-datatable":
        selected_features = pd.DataFrame(features_rows).iloc[selected_features_id]
        selected_features = selected_features["Variable"].to_list()

        cols = [col for col in df_data_model.columns if col in selected_features]
        df_data_model = df_data_model[cols]
        
        selected_features = list(set(selected_features) - set(df_data_model.columns))
        df_selected = df[selected_features]
        
        df_data_model = pd.concat([df_data_model, df_selected], axis=1, sort=False)

        impute_table = update_impute_table(df_data_model)

        # impute_figures = update_impute_figures(df_data_model)

        result = [dash.no_update] * 7 + impute_table + ['']
        # result = [dash.no_update] * 7 + impute_table + impute_figures
        # result = [dash.no_update] * 11
    elif button_id == "impute-dataset-radioitems":
        # print(df_data_model)
        if impute_dataset == 0:
            data = df_data_model
        elif impute_dataset == 1:
            data = df_data_target
        
        impute_table = update_impute_table(data)
        impute_figures = update_impute_figures(data)
    
        result = [dash.no_update] * 7 + impute_table + impute_figures
    elif button_id == "impute_apply_button" and impute_apply_clicks and selected_features_id is not None:
        # if df_data_model is None:
        #     selected_features = pd.DataFrame(features_rows).iloc[selected_features_id]
        #     selected_features = selected_features["Variable"].to_list()
        #     df_selected = df[selected_features]
        #     df_data_model = df_selected
            
        impute_methods = [knn_mean, interpolate, interpolate, interpolate, periods_mean]
        if impute_selected == 5:
            df_tmp = pd.concat([df_data_model, df_data_target], axis=1, sort=False)
            df_tmp = df_tmp.dropna()
            print(df_tmp.isnull().sum())

            df_data_model = df_tmp[df_tmp.columns.difference(["PM10"])]
            df_data_target = df_tmp[["PM10"]]

            impute_table = update_impute_table(df_data_model)
            impute_figures = update_impute_figures(df_data_model)
        else:
            if impute_selected == 1:
                impute_parameter = "linear"
            elif impute_selected == 2:
                impute_parameter = "quadratic"
            elif impute_selected == 3:
                impute_parameter = "cubic"

            if impute_dataset == 0:
                for column in df_data_model:
                    df_data_model[column] = impute_methods[impute_selected](df_data_model[column], impute_parameter)

                impute_table = update_impute_table(df_data_model)
                impute_figures = update_impute_figures(df_data_model)
            elif impute_dataset == 1:
                for column in df_data_target:
                    df_data_target[column] = impute_methods[impute_selected](df_data_target[column], impute_parameter)

                impute_table = update_impute_table(df_data_target)
                impute_figures = update_impute_figures(df_data_target)
        # print(df_data_model)

        result = [dash.no_update] * 7 + impute_table + impute_figures

    elif button_id == "impute_reset_button":
        selected_features = pd.DataFrame(features_rows).iloc[selected_features_id]
        selected_features = selected_features["Variable"].to_list()
        df_selected = df[selected_features]
        df_data_model = df_selected

        impute_table = update_impute_table(df_data_model)

        # impute_figures = update_impute_figures(df_data_model)

        result = [dash.no_update] * 7 + impute_table + ['']

    return result

def grid_search(X, y, date_split,
                min_depth, max_depth,
                min_child_weight, max_child_weight,
                min_eta, max_eta,
                min_subsample, max_subsample,
                min_colsample_bytree, max_colsample_bytree,
                eval_metric, result=None):
    X_train = X.loc[:date_split].values
    y_train = y.loc[:date_split].values

    X_test = X.loc[date_split:].values
    y_test = y.loc[date_split:].values

    dtrain = xgboost.DMatrix(X_train,y_train)
    dtest = xgboost.DMatrix(X_test,y_test)

    num_boost_round = 999
    early_stopping_rounds = 10
    nfold = 5

    if eval_metric == 0:
        eval_metric = "rmse"
    elif eval_metric == 1:
        eval_metric = "rmsle"
    elif eval_metric == 2:
        eval_metric = "mae"

    params = {
        # Parametros fijos
        'objective' : 'reg:squarederror',
        'eval_metric' : eval_metric
    }

    if min_depth == max_depth and min_child_weight == max_child_weight:
        params['max_depth'] = min_depth
        params['min_child_weight'] = min_child_weight
    else:
        gridsearch_params = [
            (depth, child_weight)
            for depth in range(min_depth, max_depth + 1)
            for child_weight in range(min_child_weight, max_child_weight + 1)
        ]

        min_mae = float("Inf")
        best_params = None
        for depth, child_weight in gridsearch_params:
            print("CV with max_depth={}, min_child_weight={}".format(
                                depth,
                                child_weight))    # Update our parameters
            params['max_depth'] = depth
            params['min_child_weight'] = child_weight
            cv_results = xgboost.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                # seed=42,
                nfold=nfold,
                metrics={eval_metric},
                early_stopping_rounds=early_stopping_rounds
            )
            mean_mae = cv_results[f"test-{eval_metric}-mean"].min()
            boost_rounds = cv_results[f"test-{eval_metric}-mean"].argmin()
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (depth, child_weight)

        params['max_depth'] = best_params[0]
        params['min_child_weight'] = best_params[1]

    print('GRID SEARCH 1')

    if min_subsample == max_subsample and min_colsample_bytree == max_colsample_bytree:
        params['subsample'] = min_subsample
        params['colsample_bytree'] = min_colsample_bytree
    else:
        gridsearch_params = [
            (subsample, colsample)
            for subsample in [round(i,1) for i in np.arange(min_subsample, max_subsample + 0.01, 0.1)]
            for colsample in [round(i,1) for i in np.arange(min_colsample_bytree, max_colsample_bytree + 0.01, 0.1)]
        ]

        min_mae = float("Inf")
        best_params = None
        for subsample, colsample in reversed(gridsearch_params):
            print("CV with subsample={}, colsample={}".format(
                                subsample,
                                colsample))
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample    # Run CV
            cv_results = xgboost.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                # seed=42,
                nfold=nfold,
                metrics={eval_metric},
                early_stopping_rounds=early_stopping_rounds
            )
            mean_mae = cv_results[f"test-{eval_metric}-mean"].min()
            boost_rounds = cv_results[f"test-{eval_metric}-mean"].argmin()
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (subsample,colsample)

        params['subsample'] = best_params[0]
        params['colsample_bytree'] = best_params[1]

    print('GRID SEARCH 2')

    if min_eta == max_eta:
        params['eta'] = min_eta
    else:

        gridsearch_params = [
            eta
            for eta in [round(i,1) for i in np.arange(min_eta, max_eta + 0.01, 0.1)]
        ]

        min_mae = float("Inf")
        best_params = None
        for eta in reversed(gridsearch_params):
            print("CV with eta={}".format(eta))    # We update our parameters
            params['eta'] = eta    # Run and time CV
        #     %time cv_results = xgboost.cv(params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=nfold, metrics=['mae'], early_stopping_rounds=early_stopping_rounds)    # Update best score
            cv_results = xgboost.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                # seed=42,
                nfold=nfold,
                metrics={eval_metric},
                early_stopping_rounds=early_stopping_rounds
            )    # Update best score
            mean_mae = cv_results[f"test-{eval_metric}-mean"].min()
            boost_rounds = cv_results[f"test-{eval_metric}-mean"].argmin()
        #     print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = eta

        params['eta'] = best_params

    print('GRID SEARCH 3')

    model = xgboost.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=early_stopping_rounds
    )

    print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

    num_boost_round = model.best_iteration + 1
    best_model = xgboost.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")]
    )

    print(best_model)

    y_pred = best_model.predict(dtest)

    mse = mean_squared_error(y_test, y_pred, squared=False)
    msle = mean_squared_log_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    expl_var = explained_variance_score(y_test, y_pred)

    # result = [dash.no_update] * 4

    model_columns = [{'id': "Parametro", 'name': "Parametro"},
                     {'id': "Valor", 'name': "Valor"}]

    # best_model_params = best_model.get_params()

    n_params = len(params)

    model_data = [None] * n_params

    for i in range(n_params):
        model_data[i] = {"Parametro":list(params.keys())[i],
                         "Valor":list(params.values())[i]}

    validate_columns = [{'id': "Criterio", 'name': "Criterio"},
                        {'id': "Valor", 'name': "Valor"}]
    
    validate_data = [{'Criterio': "MSE", 'Valor': mse},
                     {'Criterio': "MSLE", 'Valor': msle},
                     {'Criterio': "MAE", 'Valor': mae},
                     {'Criterio': "R2", 'Valor': r2},
                     {'Criterio': "Varianza explicada", 'Valor': expl_var}]

    style_data_conditional = []

    style_data_conditional.append({'if': {'state': 'active'},
                                   'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                   'border': '1px solid rgb(0, 116, 217)'})

    # print(y_test)

    validate_graph = pd.DataFrame()
    validate_graph["real"] = y_test
    validate_graph["pred"] = y_pred
    validate_graph.set_index(df_data_model.loc[date_split:].index, inplace=True)
    
    fig = validate_graph.plot()
    fig['layout']['xaxis']['type'] = 'category'

    validate_graphs = html.Div([html.Div(dcc.Graph(figure=fig))])

    # result.append(columns)
    # result.append(data)
    # result.append(style_data_conditional)
    # result.append(graphs)
    # print(result)

    result = [model_columns, model_data, style_data_conditional,
              validate_columns, validate_data, style_data_conditional, validate_graphs]

    return result

@app.callback(
    [Output('model-datatable', 'columns'),
     Output('model-datatable', 'data'),
     Output('model-datatable', 'style_data_conditional'),
     Output('validate-datatable', 'columns'),
     Output('validate-datatable', 'data'),
     Output('validate-datatable', 'style_data_conditional'),
     # Graphs
     Output('validate-graph', 'children'),
     ],
    [Input('model_apply_button', 'n_clicks'),],
    [State('model-date-split', 'date'),
     State("model-min-depth-parameter", "value"),
     State("model-max-depth-parameter", "value"),
     State("model-min-child-weight-parameter", "value"),
     State("model-max-child-weight-parameter", "value"),
     State("model-min-eta-parameter", "value"),
     State("model-max-eta-parameter", "value"),
     State("model-min-subsample-parameter", "value"),
     State("model-max-subsample-parameter", "value"),
     State("model-min-colsample-bytree-parameter", "value"),
     State("model-max-colsample-bytree-parameter", "value"),
     State("model-eval-metric-radioitems", "value")],
)
def model(model_click, date_split,
          min_depth, max_depth,
          min_child_weight, max_child_weight,
          min_eta, max_eta,
          min_subsample, max_subsample,
          min_colsample_bytree, max_colsample_bytree,
          eval_metric):
    global df_data_model, df_data_target

    print("MODEL")

    if df_data_model.empty or df_data_target.empty:
        print(df_data_model)
        print(df_data_target)
        raise dash.exceptions.PreventUpdate
        return

    X = df_data_model
    y = df_data_target['PM10']

    result = grid_search(X, y, date_split,min_depth, max_depth,
                                          min_child_weight, max_child_weight,
                                          min_eta, max_eta,
                                          min_subsample, max_subsample,
                                          min_colsample_bytree, max_colsample_bytree,
                                          eval_metric)

    # manager = multiprocessing.Manager()
    # result = manager.list()

    # p = multiprocessing.Process(target=grid_search, args=(X, y, date_split,min_depth, max_depth,
    #                                       min_child_weight, max_child_weight,
    #                                       min_eta, max_eta,
    #                                       min_subsample, max_subsample,
    #                                       min_colsample_bytree, max_colsample_bytree,
    #                                       eval_metric, result))  # sleep for 15 seconds
    # p.start()                               # but do not join

    # p.join()

    # print(result)

    return result


@app.callback(
    [Output("collapse-overview", "is_open"),
     Output("collapse-features-selection", "is_open"),
     Output("collapse-impute", "is_open"),
     Output("collapse-model", "is_open"),
     Output("collapse-validate", "is_open")],
    [Input('filter_button', 'n_clicks'),
     Input('stations', 'value'),
     Input("frequency-radioitems",'value'),
     Input("overview-toggle", "n_clicks"),
     Input("features-selection-toggle", "n_clicks"),
     Input("impute-toggle", "n_clicks"),
     Input("model-toggle", "n_clicks"),
     Input("validate-toggle", "n_clicks")],
    [State("collapse-overview", "is_open"),
     State("collapse-features-selection", "is_open"),
     State("collapse-impute", "is_open"),
     State("collapse-model", "is_open"),
     State("collapse-validate", "is_open")],
)
def toggle_accordion(filter_click, selected_station, selected_frequency,
                     overview_click, features_selection_click, impute_click, model_click, validate_click,
                     is_open_overview, is_open_features_selection, is_open_impute, is_open_model, is_open_validate):

    ctx = dash.callback_context
    if not ctx.triggered:
        return False, False, False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "overview-toggle" and overview_click:
        is_open_overview = not is_open_overview
    elif button_id == "features-selection-toggle" and features_selection_click:
        is_open_features_selection = not is_open_features_selection
    elif button_id == "impute-toggle" and impute_click:
        is_open_impute = not is_open_impute
    elif button_id == "model-toggle" and model_click:
        is_open_model = not is_open_model
    elif button_id == "validate-toggle" and validate_click:
        is_open_validate = not is_open_validate
    elif button_id == "filter_button":
        is_open_overview = True
        is_open_features_selection = True
        is_open_impute = True
        is_open_model = True
        is_open_validate = True
    elif button_id == "stations" or button_id == "frequency-radioitems":
        is_open_overview = False
        is_open_features_selection = False
        is_open_impute = False
        is_open_model = False
        is_open_validate = False
    return is_open_overview, is_open_features_selection, is_open_impute, is_open_model, is_open_validate

@app.callback(
    [Output("impute-label", "children"),
     Output('impute-label', 'style'),
     Output('impute-parameter', 'style'),
     Output('impute-parameter', 'value'),
     Output('impute-dataset-label', 'style'),
     Output('impute-dataset-radioitems', 'style'),],
    [Input("impute-radioitems", "value"),],
)
def on_form_change(radio_checked):
    text =  ""
    impute_method_style = {'display': 'none'}
    impute_method_value = None
    impute_dataset_style = {'display': 'block'}
    if radio_checked == 0:
        text = "Ingrese la cantidad de vecinos a considerar"
        impute_method_style = {'display': 'block'}
    # elif radio_checked == 5:
    #     impute_dataset_style = {'display': 'none'}
    return [text, impute_method_style, impute_method_style, impute_method_value, impute_dataset_style, impute_dataset_style]

# @app.callback(
#     [Output('features-datatable', 'columns'),
#      Output('features-datatable', 'data'),
#      Output('features-datatable', 'style_data_conditional')],
#     [Input('filter_button', 'n_clicks')],
#     [State('stations', 'value')])

# def update_features_table(n_clicks, selected_station):
#     if selected_station is None:
#         raise dash.exceptions.PreventUpdate

#     df = read_data(selected_station)

#     df_table = data_table(df)

#     # "deletable": True
#     # columns = [{'id': c, 'name': c} for c in df_table.columns if c != 'id']
#     columns = [{'id': "Variable", 'name': "Variable"}]

#     data = df_table[["Variable"]].to_dict('records')

#     print(data)

#     style_data_conditional = []

#     style_data_conditional.append({'if': {'state': 'active'},
#                                    'backgroundColor': 'rgba(0, 116, 217, 0.3)',
#                                    'border': '1px solid rgb(0, 116, 217)'})

#     print("TABLA FEATURES")

#     return [columns, data, style_data_conditional]