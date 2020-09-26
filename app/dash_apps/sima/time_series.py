# Dash
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
# Plotly
import plotly.graph_objects as go
# Pandas
import pandas as pd
# Files System Paths
from os import path

folder_name = "input"
file_name = "data_daily_avg.csv"
data_path = path.join(path.dirname(__file__), folder_name, file_name)

df = pd.read_csv(data_path, header=[0,1,2], index_col=0, parse_dates=True)
df.index.freq = 'D'

all_variables = df.columns.get_level_values(0).unique()
variables = [{"label":v, "value":v} for v in all_variables]
variables.insert(0, {"label":"All", "value":"All"})

all_stations = df.columns.get_level_values(2).unique()
stations = [{"label":s, "value":s} for s in all_stations]
stations.insert(0, {"label":"All", "value":"All"})

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.GRID]

app = DjangoDash(__name__.split(".")[-1], serve_locally=True, add_bootstrap_links=True)

filter_variables = dbc.FormGroup(
    [
        dbc.Label("Variables: "),
        dcc.Dropdown(
            id='variables',
            options=variables,
            multi=True,
            value=None,
            # placeholder="Selecciona las variables a graficar",
        ),
    ]
)

filter_stations = dbc.FormGroup(
    [
        dbc.Label("Estaciones: "),
        dcc.Dropdown(
            id='stations',
            options=stations,
            multi=True,
            value=None,
            # placeholder="Selecciona las estaciones a graficar",
        ),
    ]
)

filter_year_slider = dbc.FormGroup(
    [
        dbc.Label("Periodo de tiempo"),
        dcc.RangeSlider(
            id='year_slider',
            min=df.index.year.min(),
            max=df.index.year.max(),
            value=[df.index.year.max() - 2, df.index.year.max()],
            marks={str(year): str(year) for year in df.index.year.unique()},
            # included=False
            step=None,
        ),
    ]
)

filter_graph_type = dbc.FormGroup(
    [
        dbc.Label("Tipo de gráfica", html_for="graph_types"),
        dbc.RadioItems(
            id="graph_types",
            options=[
                {"label": "Caja", "value": 1},
                {"label": "Serie", "value": 2},
            ],
            value=1,
            inline=True,
        ),
    ],
    inline=True,
    # row=True,
)

filter_button = dbc.Button(
    "Filtrar",
    id="filter_button",
    outline=True,
    color="primary",
    className="mr-1"
)

filter_message = dbc.Alert(
    "Es necesario llenar todos los filtros.",
    id="alert_fade",
    dismissable=True,
    is_open=False,
    color="danger",
)

html_filters = dbc.Card(
    [
        filter_variables,
        filter_stations,
        filter_year_slider,
        filter_graph_type,
        filter_button,
        filter_message,
    ],
    body=True,
    style={"position":"sticky", "top":0},
)

html_graphs = html.Div(
    [
        dcc.Loading(
            id="loading",
            children=
            [
                html.Div(id='graphs'),
            ],
            type="graph",
            fullscreen=True,
        )
    ],
),

app.layout = dbc.Container(
    [
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html_filters, md=3),
                dbc.Col(html_graphs, md=9),
            ],
            # align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    [
        Output('loading', 'children'),
        Output("alert_fade", "is_open"),
    ],
    [
        Input('filter_button', 'n_clicks'),
    ],
    [
        State('variables', 'value'),
        State('stations', 'value'),
        State('year_slider', 'value'),
        State("graph_types", "value"),
        State("alert_fade", "is_open"),
    ],
)

def update_figure(n_clicks, variables, stations, selected_year, selected_graph, is_open):
    figs=[]

    if variables and stations and selected_year:
        filtered_df = df[str(selected_year[0]) : str(selected_year[1])]
        if "ALL" in str(variables).upper():
            variables = filtered_df.columns.get_level_values(0).unique()
        if "ALL" in str(stations).upper():
            stations = filtered_df.columns.get_level_values(2).unique()

        for v in variables:
            fig = go.Figure()
            m = filtered_df[v].columns.get_level_values(0).unique()[0]
            for s in stations:
                if selected_graph == 1:
                    trace = go.Box(y=filtered_df[v][m, s],
                                   name=s)
                elif selected_graph == 2:
                    trace = go.Scatter(x=filtered_df.index,
                                       y=filtered_df[v][m, s],
                                       mode="lines",
                                       name=s)
                fig.add_trace(trace)

            fig.update_layout(title=dict(text=f"Variable: {v}",
                                        y=0.97,
                                        x=0.5,
                                        xanchor="center",
                                        yanchor="top"),
                            xaxis=dict(title="Tiempo",
                                        gridcolor="white",
                                        gridwidth=2,),
                            yaxis=dict(title=f"{m}",
                                        gridcolor="white",
                                        gridwidth=2,),
                            legend=dict(orientation="h",
                                        title_text="Estaciones",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                        ),
                            plot_bgcolor='rgb(243, 243, 243)',
                            paper_bgcolor='rgb(243, 243, 243)',
                            transition_duration=700,
                            )
            figs.append(html.Div(dcc.Graph(figure=fig)))
    elif n_clicks:
        is_open = True
    return [figs, is_open]