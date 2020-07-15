# Dash
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Pandas
import pandas as pd
import math

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DjangoDash('bubble', external_stylesheets=external_stylesheets)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        step=None,
        included=False
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('year-slider', 'value')])
def update_figure(selected_year):
    filtered_df = df[df.year == selected_year].sort_values(['continent', 'country'])

    hover_text = []
    bubble_size = []

    for index, row in filtered_df.iterrows():
        hover_text.append(('Country: {country}<br>'+
                        'Life Expectancy: {lifeExp}<br>'+
                        'GDP per capita: {gdp}<br>'+
                        'Population: {pop}<br>'))
        bubble_size.append(math.sqrt(row['pop']))

    filtered_df['text'] = hover_text
    filtered_df['size'] = bubble_size
    sizeref = 2. * max(filtered_df['size']) / (100 ** 2)

    # Dictionary with dataframes for each continent
    continent_names = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']
    continent_data = {continent:filtered_df.query("continent == '%s'" %continent)
                                for continent in continent_names}

    # Create figure
    fig = go.Figure()

    for continent_name, continent in continent_data.items():
        fig.add_trace(go.Scatter(
            x=continent['gdpPercap'], y=continent['lifeExp'],
            name=continent_name, text=continent['text'],
            marker_size=continent['size'],
            ))

    fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                              sizeref=sizeref, line_width=2))

    fig.update_layout(
        title='Life Expectancy v. Per Capita GDP, 2007',
        xaxis=dict(
            title='GDP per capita (2000 dollars)',
            gridcolor='white',
            type='log',
            gridwidth=2,
        ),
        yaxis=dict(
            title='Life Expectancy (years)',
            gridcolor='white',
            gridwidth=2,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
    )

    fig.update_layout(transition_duration=500)

    # fig2 = fig

    # f = make_subplots(rows=2, cols=1, subplot_titles=("First Subplot","Second Subplot"), shared_xaxes=True)

    # f.add_trace(fig, row=1, col=1)
    # f.add_trace(fig, row=1, col=1)

    return fig