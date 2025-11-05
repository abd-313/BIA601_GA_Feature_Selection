# analysis/urls.py

from django.urls import path
from . import views

# Set the application namespace
app_name = 'analysis' 

urlpatterns = [
    # Path for the main menu page
    path('', views.home_view, name='home_view'), 
    
    # Path for dynamically loading forms (e.g., CSV, Database)
    path('upload_dynamic/', views.upload_dynamic_view, name='upload_dynamic_view'),
    
    # CRITICAL CHANGE: New path for displaying PRE-CALCULATED Repository Results
    path('repository_results/', views.display_repo_results, name='display_repo_results'),
    
    # Path to view results (e.g., after custom upload)
    path('results/<str:job_id>/', views.repository_results_view, name='repository_results_view'),
    
    path('submit_job/', views.submit_dynamic_job, name='submit_dynamic_job'), 

    path('results/<str:job_id>/', views.repository_results_view, name='results_view'),
]