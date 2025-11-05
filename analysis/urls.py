# analysis/urls.py

from django.urls import path
from . import views

app_name = 'analysis' 

urlpatterns = [
    
    path('', views.home_view, name='home_view'), 
    path('upload_dynamic/', views.upload_dynamic_view, name='upload_dynamic_view'),
    path('repository_results/', views.display_repo_results, name='display_repo_results'),
    path('results/<str:job_id>/', views.repository_results_view, name='repository_results_view'),
    path('submit_job/', views.submit_dynamic_job, name='submit_dynamic_job'), 
    path('results/<str:job_id>/', views.repository_results_view, name='results_view'),
]