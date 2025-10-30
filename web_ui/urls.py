# web_ui/urls.py
from django.urls import path
from . import views # To import the view functions that will render the pages

urlpatterns = [
    # This path maps the empty URL ('') (the home page) to the 'index' function in views.py
    path('', views.index, name='index'), 
]