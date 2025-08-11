from django.urls import path
from . import views

app_name = "recs"

urlpatterns = [
    path('', views.my_recommendations, name='list'),
    path('train/', views.train_models, name='train'),
]