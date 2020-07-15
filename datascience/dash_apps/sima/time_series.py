# Dash
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DjangoDash('time_series', external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.RangeSlider(
        id='year-slider',
        min=df.index.year.min(),
        max=df.index.year.max(),
        value=[df.index.year.min(), df.index.year.min()],
        marks={str(year): str(year) for year in df.index.year.unique()},
        step=None,
        # included=False
    ),
    dcc.Graph(id='graph-with-slider'),
])

@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('year-slider', 'value')])

def update_figure(selected_year):
    variables = df.columns.get_level_values(0).unique()
    v = variables[0]
    filtered_df = df[str(selected_year[0]) : str(selected_year[1])][v]

    fig = go.Figure()
    for m in filtered_df:
        fig.add_trace(go.Scatter(x=filtered_df.index,
                                 y=filtered_df[m[0], m[1]],
                                 mode="lines",
                                 name=m[1]))

    fig.update_layout(title=dict(text=f"Variable: {v}",
                                 y=0.95,
                                 x=0.5,
                                 xanchor="center",
                                 yanchor="top"),
                      xaxis=dict(title="Tiempo",
                                 gridcolor="white",
                                 gridwidth=2,),
                      yaxis=dict(title=f"{m[0]}",
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

    return fig