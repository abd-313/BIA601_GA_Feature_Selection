# ga_project/urls.py
from django.contrib import admin
# Make sure to import include
from django.urls import path, include 

urlpatterns = [
    path('admin/', admin.site.urls),
    # This tells Django that any request starting with '/' should be directed to the web_ui app
    path('', include('web_ui.urls')), 
]