from django.urls import path
from . import views

urlpatterns = [
    # Main landing page route
    path('', views.home_view, name='home'), 
    # API route for making a prediction using the pre-trained static model
    path('api/predict_static/', views.predict_static_api, name='predict_static_api'),
    # Route for uploading a custom CSV file for long-running analysis (Future/Dynamic)
    path('upload_dynamic/', views.upload_dynamic_view, name='upload_dynamic'), 
]
