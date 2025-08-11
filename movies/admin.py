from django.contrib import admin
from .models import Movie, Genre, Rating, Watchlist

@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ('title', 'vote_average', 'popularity')
    search_fields = ('title',)
    list_filter = ('genres',)

admin.site.register(Genre)
admin.site.register(Rating)
admin.site.register(Watchlist)