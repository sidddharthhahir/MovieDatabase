from django.urls import path
from . import views

app_name = 'recs'

urlpatterns = [
    path('', views.my_recommendations, name='list'),
    path('train/', views.train_models, name='train'),
    path('watchlist/', views.my_watchlist, name='watchlist'),            # NEW
    path('watchlist/toggle/<int:movie_id>/', views.toggle_watchlist, name='watchlist_toggle'),  # NEW
]