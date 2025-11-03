"""
bia601_project URL Configuration.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Admin interface for Django
    path('admin/', admin.site.urls),
    # Directs all root requests to the 'analysis' application
    path('', include('analysis.urls')), 
]
