from django.urls import path

from . import views

app_name = 'vrp'
urlpatterns = [
    path('', views.index, name='index'),
    path('data/', views.data, name='data'),
    path('problem_settings/', views.problem_setting, name='problem_settings'),
    path('problem_solution/', views.problem_solution, name='problem_solution'),
    path('create_problem/', views.create_problem, name='create_problem'),
    path('other_settings/', views.other_setting, name='other_settings'),
    path('visualization_setting/', views.visualization_setting, name='visualization_setting'),
    path('visualization/', views.visualization, name='visualization')
]