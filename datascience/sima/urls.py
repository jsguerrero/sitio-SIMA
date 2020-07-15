from django.urls import path
from . import views
from dash_apps.sima import time_series

app_name = 'sima'
urlpatterns = [
    path('', views.overview, name='overview'),
    path('data_exploration', views.data_exploration, name='data_exploration'),
]
