from django.urls import path
from . import views

app_name = 'movies'
urlpatterns = [
    path('', views.movie_list, name='list'),
    path('<int:tmdb_id>/', views.movie_detail, name='detail'),
    path('<int:tmdb_id>/rate/', views.rate_movie, name='rate'),
    path('<int:tmdb_id>/watchlist/', views.toggle_watchlist, name='watchlist'),
]