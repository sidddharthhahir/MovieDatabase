# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# from django.contrib import messages
# from django.db.models import Q
# from movies.models import Movie, Rating
# from .ml_models import recommendation_system
# from .explainers import explainer
# import joblib, os
# from django.conf import settings

# @login_required
# def train_models(request):
#     try:
#         recommendation_system.train_content_model()
#         recommendation_system.train_collaborative_model()
#         messages.success(request, 'Models trained successfully.')
#     except Exception as e:
#         messages.error(request, f'Error training models: {e}')
#     return redirect('recs:list')

# @login_required
# def my_recommendations(request):
#     recommendation_system.load_models()
#     seed = request.GET.get('seed')
#     recs = []
#     explain_cards = []
#     user_id = request.user.id

#     # If user has no ratings yet, suggest popular
#     if not Rating.objects.filter(user_id=user_id).exists():
#         movies = Movie.objects.order_by('-popularity')[:12]
#         for m in movies:
#             explain_cards.append({
#                 'title': m.title,
#                 'why': "Popular and highly rated by the community.",
#                 'prob': None,
#                 'shap_img': None,
#                 'lime_list': None,
#                 'movie': m,
#             })
#         return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})

#     # Otherwise try hybrid: if seed provided, content-based; plus collaborative explanation if available
#     if seed:
#         seed = int(seed)
#         content_recs = recommendation_system.get_content_recommendations(seed, n_recommendations=10)
#         movie_map = {m.id: m for m in Movie.objects.filter(id__in=[r['movie_id'] for r in content_recs])}
#         for r in content_recs:
#             m = movie_map.get(r['movie_id'])
#             if not m: continue
#             content_expl = explainer.explain_content_recommendation(seed, m.id)
#             why_text = explainer.generate_explanation_text(content_expl, m.title)
#             explain_cards.append({
#                 'title': m.title,
#                 'why': why_text,
#                 'prob': None,
#                 'shap_img': None,
#                 'lime_list': [{'feature': f['name'], 'importance': round(f['contribution'], 4)} for f in content_expl.get('features', [])][:6],
#                 'movie': m,
#             })

#     # Collaborative explanation (if model exists and user in matrix)
#     model_path = os.path.join(settings.BASE_DIR, 'ml_models')
#     has_collab = os.path.exists(os.path.join(model_path, 'collaborative_model.pkl'))
#     if has_collab:
#         # Use user's highly rated top-3 to anchor
#         rated = Rating.objects.filter(user_id=user_id).order_by('-rating')[:3]
#         # Use popular unseen as candidates; in a real system, combine with model predictions
#         seen_ids = Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True)
#         candidates = Movie.objects.exclude(id__in=seen_ids).order_by('-popularity')[:30]
#         for m in candidates[:10]:
#             collab_expl = explainer.explain_collaborative_recommendation(user_id, m.id)
#             if collab_expl.get('error'):
#                 continue
#             prob = collab_expl.get('prediction_probability', 0)
#             shap_vals = collab_expl.get('shap_values', [])
#             feature_names = [f"Movie_{i}" for i in collab_expl.get('top_movies', [])]
#             shap_img = explainer.create_shap_plot(shap_values=__import__('numpy').array(shap_vals), feature_names=feature_names)
#             why_text = explainer.generate_explanation_text(collab_expl, m.title)
#             explain_cards.append({
#                 'title': m.title,
#                 'why': why_text,
#                 'prob': f"{prob:.1%}",
#                 'shap_img': shap_img,
#                 'lime_list': collab_expl.get('lime_features', [])[:6],
#                 'movie': m,
#             })

#     if not explain_cards:
#         movies = Movie.objects.order_by('-popularity')[:12]
#         for m in movies:
#             explain_cards.append({
#                 'title': m.title,
#                 'why': "Fallback to popularity since no model/explanations available yet.",
#                 'prob': None,
#                 'shap_img': None,
#                 'lime_list': None,
#                 'movie': m,
#             })

#     return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from movies.models import Movie, Rating
from django.conf import settings
import os
import numpy as np
from .ml_models import recommendation_system
from .explainers import explainer
from recommendations.utils.feature_mapping import feature_to_movie, load_user_movie_columns
import re

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
    seed = request.GET.get('seed')
    explain_cards = []
    user_id = request.user.id

    # Cold-start: Popular movies
    if not Rating.objects.filter(user_id=user_id).exists():
        movies = Movie.objects.order_by('-popularity')[:12]
        for m in movies:
            explain_cards.append({
                'title': m.title,
                'why': "Popular and highly rated by the community.",
                'prob': None,
                'shap_img': None,
                'lime_list': None,
                'movie': m,
            })
        return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})

    # Content-based seed flow
    if seed:
        seed = int(seed)
        content_recs = recommendation_system.get_content_recommendations(seed, n_recommendations=10)
        movie_map = {m.id: m for m in Movie.objects.filter(id__in=[r['movie_id'] for r in content_recs])}
        for r in content_recs:
            m = movie_map.get(r['movie_id'])
            if not m:
                continue
            content_expl = explainer.explain_content_recommendation(seed, m.id)
            why_text = explainer.generate_explanation_text(content_expl, m.title)
            explain_cards.append({
                'title': m.title,
                'why': why_text,
                'prob': None,
                'shap_img': None,
                'lime_list': [{'feature': f['name'], 'importance': round(f['contribution'], 4)} for f in content_expl.get('features', [])][:6],
                'movie': m,
            })

    # Collaborative explanations (with item-aware charts)
    model_path = os.path.join(settings.BASE_DIR, 'ml_models')
    has_collab = os.path.exists(os.path.join(model_path, 'collaborative_model.pkl'))
    if has_collab:
        user_ratings_map = dict(Rating.objects.filter(user_id=user_id).values_list('movie_id', 'rating'))
        seen_ids = Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True)
        candidates = Movie.objects.exclude(id__in=seen_ids).order_by('-popularity')[:30]

        for m in candidates[:10]:
            collab_expl = explainer.explain_collaborative_recommendation(user_id, m.id)
            if collab_expl.get('error'):
                continue

            prob = collab_expl.get('prediction_probability', 0.0)
            shap_vals = collab_expl.get('shap_values', []) or []

            # Item-aware contributions for the candidate (varies per card)
            item_aware = explainer.explain_item_aware_with_content(
                user_id=user_id,
                candidate_movie_id=m.id,
                content_model=getattr(recommendation_system, "content_model", None),
                top_k=10,
                rating_baseline=3.0
            )

            if item_aware and item_aware.get("item_contribs"):
                chart_names = item_aware["feature_names"]
                chart_vals = item_aware["shap_values"]
                shap_img = explainer.create_shap_plot(
                    shap_values=np.array(chart_vals),
                    feature_names=chart_names,
                    max_display=10,
                    sort_values=False
                )
                because_titles = [c["title"] for c in item_aware.get("item_contribs", []) if c["weight"] > 0]
                because_sentence = f"Because you liked {', '.join(because_titles[:3])}" if because_titles else ""
            else:
                # Fallback to user-global SHAP but label with real titles
                names = collab_expl.get('feature_titles')
                if not names:
                    cols = load_user_movie_columns()
                    names = []
                    for i in (collab_expl.get('top_movies', []) or []):
                        _, t = feature_to_movie(f"Movie_{i}", cols)
                        names.append(t or f"Movie_{i}")
                shap_img = explainer.create_shap_plot(
                    shap_values=np.array(shap_vals),
                    feature_names=names,
                    max_display=10,
                    sort_values=True
                )
                because_sentence = ""

            # Humanized LIME list
            lime_raw = collab_expl.get('lime_features', [])
            lime_human = humanize_lime_factors(lime_raw, user_ratings_map)

            why_text = explainer.generate_explanation_text(collab_expl, m.title)
            final_why = because_sentence or why_text

            explain_cards.append({
                'title': m.title,
                'why': final_why,
                'prob': f"{prob:.1%}",
                'shap_img': shap_img,
                'lime_list': lime_human,
                'movie': m,
            })

    if not explain_cards:
        movies = Movie.objects.order_by('-popularity')[:12]
        for m in movies:
            explain_cards.append({
                'title': m.title,
                'why': "Fallback to popularity since no model/explanations available yet.",
                'prob': None,
                'shap_img': None,
                'lime_list': None,
                'movie': m,
            })

    return render(request, 'recommendations/recommendations.html', {'cards': explain_cards})