import re
import joblib
import os
from django.conf import settings
from movies.models import Movie

MOVIE_FEA_RE = re.compile(r"^Movie_(\d+)$")

def load_user_movie_columns():
    """
    Loads the user-movie matrix columns (movie IDs), so we can map feature indices -> movie_id -> title.
    Assumes your user_movie_matrix.pkl has columns being Movie IDs (ints).
    """
    p = os.path.join(settings.BASE_DIR, 'ml_models', 'user_movie_matrix.pkl')
    um = joblib.load(p)  # pandas DataFrame
    return list(um.columns)  # [movie_id, movie_id, ...]

def feature_to_movie(obj, cols=None):
    """
    Accepts a feature label like 'Movie_55' (string) or (feature, value) tuple.
    Returns (movie_id, title) or (None, None) if unknown.
    """
    if isinstance(obj, tuple) and len(obj) >= 1:
        feat = obj[0]
    else:
        feat = obj

    m = MOVIE_FEA_RE.match(str(feat))
    if not m:
        return (None, None)

    idx = int(m.group(1))
    cols = cols or load_user_movie_columns()
    if idx < 0 or idx >= len(cols):
        return (None, None)

    movie_id = cols[idx]
    mv = Movie.objects.filter(id=movie_id).only('title').first()
    return (movie_id, (mv.title if mv else f"Movie {movie_id}"))