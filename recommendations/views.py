from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse
from movies.models import Movie, Rating
from django.conf import settings
from django.db import models
import os
import numpy as np
from .ml_models import recommendation_system
from .explainers import explainer
from recommendations.utils.feature_mapping import feature_to_movie, load_user_movie_columns
import re

# Optional Watchlist model (if your project has one)
try:
    from movies.models import Watchlist as WLModel
except Exception:
    WLModel = None

LIME_RE = re.compile(r"^(Movie_\d+)\s*([<>]=?)\s*([-+]?\d*\.?\d+):\s*([-+]?\d*\.?\d+)$")

def humanize_lime_factors(lime_lines, user_ratings_by_movie_id):
    cols = load_user_movie_columns()
    out = []
    for entry in (lime_lines or []):
        feat = None
        op = ">"
        thr = 0.0
        weight = None

        if isinstance(entry, str):
            m = LIME_RE.match(entry.strip())
            if not m:
                continue
            feat, op, thr_str, weight_str = m.groups()
            thr = float(thr_str)
            weight = float(weight_str)
        elif isinstance(entry, dict):
            feat = entry.get("feature") or entry.get("name")
            op = entry.get("operator", op)
            thr = float(entry.get("threshold", thr))
            if "weight" in entry:
                weight = float(entry["weight"])
            elif "importance" in entry:
                weight = float(entry["importance"])
            elif "contribution" in entry:
                weight = float(entry["contribution"])
            elif "value" in entry:
                weight = float(entry["value"])
            else:
                continue
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            feat = entry[0]
            try:
                weight = float(entry[1])
            except Exception:
                continue
        else:
            continue

        mid, title = feature_to_movie(feat, cols)
        if mid is None:
            continue

        user_rating = user_ratings_by_movie_id.get(mid)
        out.append({
            "title": title,
            "operator": op,
            "threshold": float(thr),
            "weight": float(weight),
            "user_rating": float(user_rating) if user_rating is not None else None,
        })

    # Highest absolute contributions first
    out.sort(key=lambda d: abs(d["weight"]), reverse=True)
    return out

def humanize_shap_top_features(shap_top_features, user_ratings_by_movie_id, candidate_movie=None):
    cols = load_user_movie_columns()
    title_contribs = []
    positive_titles = []

    candidate_genres = set(g.name for g in candidate_movie.genres.all()) if candidate_movie else None

    for feat, contrib in (shap_top_features or []):
        mid, title = feature_to_movie(feat, cols)
        if mid is None:
            continue
        ur = user_ratings_by_movie_id.get(mid)
        title_contribs.append((title, contrib, ur))
        if contrib > 0 and (ur is not None and ur >= 3.5):
            if candidate_genres is not None:
                mv = Movie.objects.filter(id=mid).first()
                mv_genres = set(g.name for g in mv.genres.all()) if mv else set()
                if mv_genres & candidate_genres:
                    positive_titles.append(title)
            else:
                positive_titles.append(title)

    if candidate_genres is not None and not positive_titles:
        positive_titles = [t for (t, c, ur) in title_contribs if c > 0 and (ur is not None and ur >= 3.5)]

    nice = ", ".join(positive_titles[:3]) if positive_titles else ""
    because_sentence = f"Because you liked {nice}" if nice else ""
    return title_contribs, because_sentence

@login_required
def train_models(request):
    try:
        recommendation_system.train_content_model()
        recommendation_system.train_collaborative_model()
        messages.success(request, 'Models trained successfully.')
    except Exception as e:
        messages.error(request, f'Error training models: {e}')
    return redirect('recs:list')

@login_required
def my_recommendations(request):
    recommendation_system.load_models()
    user_id = request.user.id
    explain_cards = []
    
    # Content-based seed flow (accepts either internal id or tmdb_id)
    seed_param = request.GET.get('seed')
    if seed_param:
        try:
            seed_raw = int(seed_param)
        except ValueError:
            seed_raw = None

        seed_movie = None
        if seed_raw is not None:
            # Try tmdb_id first (likely from detail page), then internal id
            seed_movie = Movie.objects.filter(tmdb_id=seed_raw).first() or Movie.objects.filter(id=seed_raw).first()

        if seed_movie:
            seed_id = seed_movie.id  # internal id
            content_recs = recommendation_system.get_content_recommendations(seed_id, n_recommendations=10)
            movie_map = {m.id: m for m in Movie.objects.filter(id__in=[r['movie_id'] for r in content_recs])}
            seen_movie_ids = set()  # de-dup in seed flow
            for r in content_recs:
                m = movie_map.get(r['movie_id'])
                if not m or m.id in seen_movie_ids:
                    continue
                seen_movie_ids.add(m.id)

                content_expl = explainer.explain_content_recommendation(seed_id, m.id)
                why_text = f"Similar to '{seed_movie.title}' - {explainer.generate_explanation_text(content_expl, m.title)}"
                # LIME features to percent strings
                lime_list = [
                    {
                        'feature': f.get('name') or f.get('feature') or '',
                        'importance': f"{abs(f.get('contribution', 0.0)) * 100:.1f}%"
                    }
                    for f in (content_expl.get('features', []) or [])
                ][:6]

                explain_cards.append({
                    'title': m.title,
                    'why': why_text,
                    'prob': f"{(r.get('score') or 0):.1%}",  # show as percent
                    'shap_img': None,
                    'lime_list': lime_list,
                    'movie': m,
                    'algorithm': 'content',
                    'score': r.get('score'),
                })
            
            return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})
    
    # User-specific recommendations
    user_recs = recommendation_system.get_user_recommendations(user_id, n_recommendations=12)

    if not user_recs:
        # Fallback to popular movies
        movies = Movie.objects.order_by('-popularity')[:12]
        for m in movies:
            explain_cards.append({
                'title': m.title,
                'why': "Popular and highly rated by the community.",
                'prob': None,
                'shap_img': None,
                'lime_list': None,
                'movie': m,
                'algorithm': 'popularity',
                'score': None,
            })
        return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})
    
    # Stabilize ordering to avoid flicker for near-equal scores
    try:
        user_recs.sort(key=lambda r: (round(float(r.get('score', 0) or 0), 6), int(r['movie_id'])), reverse=True)
    except Exception:
        pass
    
    # Fetch movie objects
    movie_ids = [rec['movie_id'] for rec in user_recs]
    movies = {m.id: m for m in Movie.objects.filter(id__in=movie_ids)}
    
    # User's ratings for explanations
    user_ratings_map = dict(Rating.objects.filter(user_id=user_id).values_list('movie_id', 'rating'))
    
    # Avoid duplicates on page
    seen_movie_ids = set()

    # Build cards
    for rec in user_recs:
        mid = rec['movie_id']
        if mid in seen_movie_ids:
            continue
        seen_movie_ids.add(mid)

        movie = movies.get(mid)
        if not movie:
            continue
        
        algorithm = rec.get('algorithm', 'popularity')
        
        if 'content' in algorithm:
            # Content-based explanation
            seed_movies = Rating.objects.filter(
                user_id=user_id, 
                rating__gte=4
            ).order_by('-rating')[:3]
            
            if seed_movies:
                seed_movie = seed_movies[0].movie
                content_expl = explainer.explain_content_recommendation(seed_movie.id, movie.id)
                why_text = f"Because you liked '{seed_movie.title}' and this movie has similar content"
                lime_list = [
                    {
                        'feature': f.get('name') or f.get('feature') or '',
                        'importance': f"{abs(f.get('contribution', 0.0)) * 100:.1f}%"
                    }
                    for f in (content_expl.get('features', []) or [])
                ][:6]
            else:
                why_text = "Based on content similarity with movies you might like"
                lime_list = []
            
            explain_cards.append({
                'title': movie.title,
                'why': why_text,
                'prob': f"{(rec.get('score') or 0):.1%}",
                'shap_img': None,
                'lime_list': lime_list,
                'movie': movie,
                'algorithm': 'content',
                'score': rec.get('score'),
            })
        
        elif 'genre' in algorithm:
            # Genre-based explanation
            user_genres = Rating.objects.filter(
                user_id=user_id, 
                rating__gte=4
            ).values_list('movie__genres__name', flat=True).distinct()
            
            movie_genres = [g.name for g in movie.genres.all()]
            common_genres = set(user_genres) & set(movie_genres)
            
            if common_genres:
                genre_str = ', '.join(list(common_genres)[:2])
                why_text = f"Because you enjoy {genre_str} movies"
            else:
                why_text = "Based on your genre preferences"
            
            explain_cards.append({
                'title': movie.title,
                'why': why_text,
                'prob': f"{(rec.get('score') or 0):.1%}",
                'shap_img': None,
                'lime_list': [],
                'movie': movie,
                'algorithm': 'genre',
                'score': rec.get('score'),
            })
        
        elif 'collaborative' in algorithm:
            # Collaborative filtering explanation
            collab_expl = explainer.explain_collaborative_recommendation(user_id, movie.id)
            
            if not collab_expl.get('error'):
                prob = collab_expl.get('prediction_probability', 0.0)
                
                # Item-aware chart for this specific movie
                item_aware = explainer.explain_item_aware_with_content(
                    user_id=user_id,
                    candidate_movie_id=movie.id,
                    content_model=getattr(recommendation_system, "content_model", None),
                    top_k=8,
                    rating_baseline=3.0
                )
                
                if item_aware and item_aware.get("item_contribs"):
                    chart_names = item_aware["feature_names"]
                    chart_vals = item_aware["shap_values"]
                    shap_img = explainer.create_shap_plot(
                        shap_values=np.array(chart_vals),
                        feature_names=chart_names,
                        max_display=8,
                        sort_values=False
                    )
                    
                    positive_contribs = [c for c in item_aware.get("item_contribs", []) if c["weight"] > 0]
                    if positive_contribs:
                        top_movies = [c["title"] for c in positive_contribs[:3]]
                        why_text = f"Because you liked {', '.join(top_movies)}"
                    else:
                        why_text = f"Recommended with {prob:.1%} confidence based on similar users"
                else:
                    shap_img = None
                    why_text = f"Recommended with {prob:.1%} confidence based on similar users"
                
                # Humanized LIME
                lime_raw = collab_expl.get('lime_features', [])
                lime_list = humanize_lime_factors(lime_raw, user_ratings_map)
                # Convert weights to percent strings for display
                for d in lime_list:
                    w = abs(d.get('weight') or 0)
                    d['importance'] = f"{w * 100:.1f}%"

                explain_cards.append({
                    'title': movie.title,
                    'why': why_text,
                    'prob': f"{prob:.1%}",  # percent string
                    'shap_img': shap_img,
                    'lime_list': lime_list,
                    'movie': movie,
                    'algorithm': 'collaborative',
                    'score': rec.get('score'),
                })
            else:
                explain_cards.append({
                    'title': movie.title,
                    'why': "Recommended based on user preferences",
                    'prob': f"{(rec.get('score') or 0):.1%}",
                    'shap_img': None,
                    'lime_list': [],
                    'movie': movie,
                    'algorithm': 'collaborative',
                    'score': rec.get('score'),
                })
        
        else:
            # Default explanation
            explain_cards.append({
                'title': movie.title,
                'why': f"Recommended based on {algorithm}",
                'prob': f"{(rec.get('score') or 0):.1%}",
                'shap_img': None,
                'lime_list': [],
                'movie': movie,
                'algorithm': algorithm,
                'score': rec.get('score'),
            })
    
    return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})

@login_required
def my_watchlist(request):
    user_id = request.user.id

    # Source A: Rating.watchlist = True
    rating_qs = (Rating.objects
                 .filter(user_id=user_id, watchlist=True)
                 .select_related('movie'))

    movies_map = {r.movie_id: r.movie for r in rating_qs}

    # Source B: Watchlist model rows (if model exists)
    if WLModel is not None:
        wl_qs = WLModel.objects.filter(user_id=user_id).select_related('movie')
        for wl in wl_qs:
            movies_map.setdefault(wl.movie_id, wl.movie)

    movies = list(movies_map.values())
    movies.sort(key=lambda m: (m.popularity or 0), reverse=True)

    return render(request, 'recommendations/watchlist.html', {'movies': movies})

@login_required
def toggle_watchlist(request, movie_id):
    user_id = request.user.id
    movie = get_object_or_404(Movie, id=movie_id)

    rating, created = Rating.objects.get_or_create(
        user_id=user_id,
        movie_id=movie.id,
        defaults={'rating': None, 'watchlist': True}
    )

    # If it already existed, flip the flag
    if not created:
        rating.watchlist = not bool(getattr(rating, 'watchlist', False))
        rating.save(update_fields=['watchlist', 'updated_at'])
    # If created, it is already True

    # Keep standalone Watchlist table in sync if it exists
    if WLModel is not None:
        if rating.watchlist:
            WLModel.objects.get_or_create(user_id=user_id, movie_id=movie.id)
        else:
            WLModel.objects.filter(user_id=user_id, movie_id=movie.id).delete()

    # AJAX support
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'ok': True, 'watchlist': bool(rating.watchlist)})

    return redirect(request.META.get('HTTP_REFERER') or reverse('recs:watchlist'))