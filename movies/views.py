# from django.shortcuts import render, get_object_or_404, redirect
# from django.contrib.auth.decorators import login_required
# from django.contrib import messages
# from .models import Movie, Rating, Watchlist
# from recommendations.ml_models import recommendation_system

# def movie_list(request):
#     q = request.GET.get('q', '')
#     genre = request.GET.get('genre')
#     movies = Movie.objects.all()
#     if q:
#         movies = movies.filter(title__icontains=q)
#     if genre:
#         movies = movies.filter(genres__name__iexact=genre)
#     movies = movies.select_related().prefetch_related('genres')[:60]
#     genres = set(g.name for g in (g for m in Movie.objects.all().prefetch_related('genres') for g in m.genres.all()))
#     return render(request, 'movies/list.html', {'movies': movies, 'q': q, 'genres': sorted(genres)})

# def movie_detail(request, tmdb_id):
#     movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
#     user_rating = None
#     if request.user.is_authenticated:
#         user_rating = Rating.objects.filter(user=request.user, movie=movie).first()
#     return render(request, 'movies/detail.html', {'movie': movie, 'user_rating': user_rating})

# # @login_required
# # def rate_movie(request, tmdb_id):
# #     movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
# #     if request.method == 'POST':
# #         try:
# #             value = float(request.POST.get('rating'))
# #             if value < 1 or value > 5:
# #                 raise ValueError
# #         except Exception:
# #             messages.error(request, 'Invalid rating.')
# #             return redirect('movies:detail', tmdb_id=tmdb_id)
# #         rating, _ = Rating.objects.update_or_create(user=request.user, movie=movie, defaults={'rating': value})
# #         messages.success(request, f'Rated {movie.title} {value} stars.')
# #     return redirect('movies:detail', tmdb_id=tmdb_id)
# @login_required
# def rate_movie(request, tmdb_id):
#     movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
#     if request.method == 'POST':
#         try:
#             value = float(request.POST.get('rating'))
#             if value < 1 or value > 5:
#                 raise ValueError
#         except Exception:
#             messages.error(request, 'Invalid rating.')
#             return redirect('movies:detail', tmdb_id=tmdb_id)

#         rating, _ = Rating.objects.update_or_create(
#             user=request.user,
#             movie=movie,
#             defaults={'rating': value}
#         )

#         # Clear user's recommendation cache so recs update immediately
#         recommendation_system.clear_user_cache(request.user.id)

#         messages.success(request, f'Rated {movie.title} {value} stars.')
#     return redirect('movies:detail', tmdb_id=tmdb_id)

# @login_required
# def toggle_watchlist(request, tmdb_id):
#     movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
#     entry = Watchlist.objects.filter(user=request.user, movie=movie).first()
#     if entry:
#         entry.delete()
#         messages.info(request, f'Removed {movie.title} from your Watchlist.')
#     else:
#         Watchlist.objects.create(user=request.user, movie=movie)
#         messages.success(request, f'Added {movie.title} to your Watchlist.')
#     return redirect('movies:detail', tmdb_id=tmdb_id)
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Movie, Rating, Watchlist
from recommendations.ml_models import recommendation_system

def movie_list(request):
    q = request.GET.get('q', '')
    genre = request.GET.get('genre')
    movies = Movie.objects.all()
    if q:
        movies = movies.filter(title__icontains=q)
    if genre:
        movies = movies.filter(genres__name__iexact=genre)
    movies = movies.select_related().prefetch_related('genres')[:60]
    genres = set(g.name for g in (g for m in Movie.objects.all().prefetch_related('genres') for g in m.genres.all()))
    return render(request, 'movies/list.html', {'movies': movies, 'q': q, 'genres': sorted(genres)})

def movie_detail(request, tmdb_id):
    movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
    user_rating = None
    if request.user.is_authenticated:
        user_rating = Rating.objects.filter(user=request.user, movie=movie).first()
    return render(request, 'movies/detail.html', {'movie': movie, 'user_rating': user_rating})

@login_required
def rate_movie(request, tmdb_id):
    movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
    if request.method == 'POST':
        try:
            value = float(request.POST.get('rating'))
            if value < 1 or value > 5:
                raise ValueError
        except Exception:
            messages.error(request, 'Invalid rating.')
            return redirect('movies:detail', tmdb_id=tmdb_id)
        
        rating, _ = Rating.objects.update_or_create(user=request.user, movie=movie, defaults={'rating': value})
        
        # Clear user's recommendation cache so recs update immediately
        recommendation_system.clear_user_cache(request.user.id)
        
        messages.success(request, f'Rated {movie.title} {value} stars.')
    return redirect('movies:detail', tmdb_id=tmdb_id)

@login_required
def toggle_watchlist(request, tmdb_id):
    movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
    entry = Watchlist.objects.filter(user=request.user, movie=movie).first()
    if entry:
        entry.delete()
        messages.info(request, f'Removed {movie.title} from your Watchlist.')
    else:
        Watchlist.objects.create(user=request.user, movie=movie)
        messages.success(request, f'Added {movie.title} to your Watchlist.')
    return redirect('movies:detail', tmdb_id=tmdb_id)