from django.urls import path
from . import views

# CRITICAL FIX: Defines the application namespace (e.g., used as {% url 'analysis:home_view' %})
app_name = 'analysis' 

urlpatterns = [
    # Home view. The name='home_view' is required for reverse lookup.
    path('', views.home_view, name='home_view'), 

    # Dynamic upload view. The name='upload_dynamic_view' is required.
    path('upload_dynamic/', views.upload_dynamic_view, name='upload_dynamic_view'),
    
    # Status view for the GA job. It must have a name and take a job_id parameter.
    path('ga_status/<str:job_id>/', views.ga_status_view, name='ga_status_view'),
]
