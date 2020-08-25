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
# Pandas
import pandas as pd
pd.options.plotting.backend = "plotly"
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
    style={"position":"sticky", "top":0},
)

html_table = dt.DataTable(
    id='datatable',
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

html_body = html.Div(
    children=[
        html.Div(
            [
                dcc.Loading(
                    id="loading1",
                    children=
                    [
                        html.Div([html_table], id='table'),
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
                    id="loading2",
                    children=
                    [
                        # html.Div([html_table], id='table'),
                        html.Div(id='graph'),
                    ],
                    type="graph",
                    # fullscreen=False,
                ),
            ],
        ),
    ],
),

app.layout = dbc.Container(
    [
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html_filters, md=3),
                dbc.Col(html_body, md=9),
            ],
            # align="center",
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
                    #EBF0F8 {max_bound_percentage}%,
                    #EBF0F8 100%)
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
                        #EBF0F8 {max_bound_percentage}%,
                        #EBF0F8 100%)
                    """.format(max_bound_percentage=max_bound_percentage)
                ),
                'paddingBottom': 2,
                'paddingTop': 2
            })
    return styles

@app.callback(
    [Output('datatable', 'columns'),
     Output('datatable', 'data'),
     Output('datatable', 'style_data_conditional')],
    [Input('filter_button', 'n_clicks')],
    [State('stations', 'value')])

def update(n_clicks, selected_station):
    if selected_station is None:
        raise dash.exceptions.PreventUpdate

    df = read_data(selected_station)

    df_table = pd.concat([df.apply(pd.Series.first_valid_index).to_frame(name="Primer Dato"),
                          df.apply(pd.Series.last_valid_index).to_frame(name="Ultimo Dato")], axis=1, sort=False)

    df_table['Espacio muestral'] = (df_table["Ultimo Dato"] - df_table["Primer Dato"]).astype('timedelta64[s]')//3600.0

    df_table['Muestras reales'] = [df_table.iloc[v, 2] - df.loc[df_table.iloc[v, 0]: df_table.iloc[v, 1], df_table.index[v]].isna().sum() for v in range(len(df_table.index))]

    df_table.index = df_table.index.set_names(['Variable'])
    df_table = df_table.reset_index(level=[0])
    df_table['id'] = df_table.index

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

    # fig = px.line(df, height=500, log_y=True)

    # fig_html = html.Div(dcc.Graph(figure=fig))

    # fig_html = html.Div()

    # print('GRAFICA')

    return [columns, data, style_data_conditional]

@app.callback(
    Output('graph', 'children'),
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
    fig = px.line(df, height=500, log_y=True)

    print('GRAFICA')

    return html.Div(dcc.Graph(figure=fig))
