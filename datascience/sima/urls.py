from django.urls import path
from . import views
from dash_apps.sima import data_overview

app_name = 'sima'
urlpatterns = [
    path('overview', views.overview, name='overview'),
    path('data_exploration', views.data_exploration, name='data_exploration'),
]
