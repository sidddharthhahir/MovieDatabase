from movies.models import Rating
try:
    from movies.models import Watchlist as WLModel
except Exception:
    WLModel = None

def watchlist_count(request):
    if not request.user.is_authenticated:
        return {}
    user_id = request.user.id
    # Count unique movies from Rating.watchlist=True
    count = Rating.objects.filter(user_id=user_id, watchlist=True).values('movie_id').distinct().count()
    # If a separate Watchlist table exists, merge without double-counting
    if WLModel is not None:
        rating_ids = set(Rating.objects.filter(user_id=user_id, watchlist=True).values_list('movie_id', flat=True))
        wl_ids = set(WLModel.objects.filter(user_id=user_id).values_list('movie_id', flat=True))
        count = len(rating_ids | wl_ids)
    return {'movies_in_watchlist_count': count}