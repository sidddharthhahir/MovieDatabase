# import shap
# import lime
# import lime.lime_tabular
# import numpy as np
# import pandas as pd
# import joblib
# import os
# from django.conf import settings
# from .ml_models import recommendation_system
# import matplotlib.pyplot as plt
# import io
# import base64
# import json

# class RecommendationExplainer:
#     def __init__(self):
#         self.model_path = os.path.join(settings.BASE_DIR, 'ml_models')
#         self.shap_explainer = None
#         self.lime_explainer = None
    
#     def initialize_explainers(self):
#         """Initialize SHAP and LIME explainers"""
#         try:
#             # Load the collaborative model if available
#             if os.path.exists(os.path.join(self.model_path, 'collaborative_model.pkl')):
#                 model = joblib.load(os.path.join(self.model_path, 'collaborative_model.pkl'))
#                 user_movie_matrix = joblib.load(os.path.join(self.model_path, 'user_movie_matrix.pkl'))
                
#                 # Initialize SHAP explainer
#                 self.shap_explainer = shap.TreeExplainer(model)
                
#                 # Initialize LIME explainer
#                 self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
#                     user_movie_matrix.values,
#                     feature_names=[f'Movie_{i}' for i in range(user_movie_matrix.shape[1])],
#                     class_names=['Dislike', 'Like'],
#                     mode='classification'
#                 )
                
#                 return True
#         except Exception as e:
#             print(f"Error initializing explainers: {e}")
        
#         return False
    
#     def explain_content_recommendation(self, movie_id, recommended_movie_id):
#         """Explain why a movie was recommended based on content similarity"""
#         try:
#             movies_df = joblib.load(os.path.join(self.model_path, 'movies_df.pkl'))
#             tfidf_vectorizer = joblib.load(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))
            
#             # Get movie details
#             original_movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
#             recommended_movie = movies_df[movies_df['movie_id'] == recommended_movie_id].iloc[0]
            
#             # Get TF-IDF features
#             original_text = original_movie['genres'] + ' ' + original_movie['overview']
#             recommended_text = recommended_movie['genres'] + ' ' + recommended_movie['overview']
            
#             # Get feature names and values
#             feature_names = tfidf_vectorizer.get_feature_names_out()
#             original_features = tfidf_vectorizer.transform([original_text]).toarray()[0]
#             recommended_features = tfidf_vectorizer.transform([recommended_text]).toarray()[0]
            
#             # Find top contributing features
#             feature_contributions = original_features * recommended_features
#             top_features_idx = np.argsort(feature_contributions)[-10:][::-1]
            
#             explanation = {
#                 'type': 'content_based',
#                 'features': [
#                     {
#                         'name': feature_names[idx],
#                         'contribution': float(feature_contributions[idx]),
#                         'original_value': float(original_features[idx]),
#                         'recommended_value': float(recommended_features[idx])
#                     }
#                     for idx in top_features_idx if feature_contributions[idx] > 0
#                 ],
#                 'similarity_factors': {
#                     'genres': original_movie['genres'],
#                     'vote_average_diff': abs(original_movie['vote_average'] - recommended_movie['vote_average']),
#                     'popularity_ratio': recommended_movie['popularity'] / max(original_movie['popularity'], 1)
#                 }
#             }
            
#             return explanation
            
#         except Exception as e:
#             print(f"Error explaining content recommendation: {e}")
#             return {'type': 'content_based', 'features': [], 'error': str(e)}
    
#     def explain_collaborative_recommendation(self, user_id, movie_id):
#         """Explain collaborative filtering recommendation using SHAP and LIME"""
#         if not self.shap_explainer or not self.lime_explainer:
#             if not self.initialize_explainers():
#                 return {'type': 'collaborative', 'error': 'Explainers not available'}
        
#         try:
#             user_movie_matrix = joblib.load(os.path.join(self.model_path, 'user_movie_matrix.pkl'))
#             model = joblib.load(os.path.join(self.model_path, 'collaborative_model.pkl'))
            
#             if user_id not in user_movie_matrix.index:
#                 return {'type': 'collaborative', 'error': 'User not found in training data'}
            
#             user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)
            
#             # SHAP explanation
#             shap_values = self.shap_explainer.shap_values(user_ratings)
#             if isinstance(shap_values, list):
#                 shap_values = shap_values[1]  # Get positive class SHAP values
            
#             # Get top contributing movies
#             feature_importance = shap_values[0]
#             top_features_idx = np.argsort(np.abs(feature_importance))[-10:][::-1]
            
#             # LIME explanation
#             lime_exp = self.lime_explainer.explain_instance(
#                 user_ratings[0], 
#                 model.predict_proba, 
#                 num_features=10
#             )
            
#             # Convert to serializable format
#             explanation = {
#                 'type': 'collaborative',
#                 'shap_values': [float(val) for val in feature_importance[top_features_idx]],
#                 'top_movies': [int(idx) for idx in top_features_idx],
#                 'lime_features': [
#                     {'feature': exp[0], 'importance': float(exp[1])} 
#                     for exp in lime_exp.as_list()
#                 ],
#                 'prediction_probability': float(model.predict_proba(user_ratings)[0][1])
#             }
            
#             return explanation
            
#         except Exception as e:
#             print(f"Error explaining collaborative recommendation: {e}")
#             return {'type': 'collaborative', 'error': str(e)}
    
#     def generate_explanation_text(self, explanation_data, movie_title):
#         """Generate human-readable explanation text"""
#         if explanation_data.get('error'):
#             return f"Unable to generate explanation: {explanation_data['error']}"
        
#         if explanation_data['type'] == 'content_based':
#             features = explanation_data.get('features', [])
#             if features:
#                 top_feature = features[0]['name']
#                 return f"'{movie_title}' was recommended because it shares similar characteristics with movies you've enjoyed, particularly in terms of '{top_feature}' and other content features."
#             else:
#                 return f"'{movie_title}' was recommended based on content similarity with your previously rated movies."
        
#         elif explanation_data['type'] == 'collaborative':
#             prob = explanation_data.get('prediction_probability', 0)
#             return f"'{movie_title}' was recommended with {prob:.1%} confidence based on preferences of users similar to you who also enjoyed movies in your rating history."
        
#         else:
#             return f"'{movie_title}' was recommended based on its popularity and general appeal."
    
#     def create_shap_plot(self, shap_values, feature_names, max_display=10):
#         """Create SHAP summary plot as base64 image"""
#         try:
#             plt.figure(figsize=(10, 6))
            
#             # Create a simple bar plot of SHAP values
#             top_indices = np.argsort(np.abs(shap_values))[-max_display:]
#             top_values = shap_values[top_indices]
#             top_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
            
#             colors = ['red' if val < 0 else 'blue' for val in top_values]
#             plt.barh(range(len(top_values)), top_values, color=colors)
#             plt.yticks(range(len(top_values)), top_names)
#             plt.xlabel('SHAP Value (Impact on Prediction)')
#             plt.title('Feature Importance for Recommendation')
#             plt.tight_layout()
            
#             # Convert to base64
#             buffer = io.BytesIO()
#             plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
#             buffer.seek(0)
#             image_base64 = base64.b64encode(buffer.getvalue()).decode()
#             plt.close()
            
#             return f"data:image/png;base64,{image_base64}"
            
#         except Exception as e:
#             print(f"Error creating SHAP plot: {e}")
#             return None

# # Initialize the explainer
# explainer = RecommendationExplainer()
import os
import io
import base64
import joblib
import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
from django.conf import settings
from .ml_models import recommendation_system
from recommendations.utils.feature_mapping import load_user_movie_columns, feature_to_movie


class RecommendationExplainer:
    def __init__(self):
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models')
        self.shap_explainer = None
        self.lime_explainer = None

    def initialize_explainers(self):
        """Initialize SHAP and LIME explainers."""
        try:
            collab_path = os.path.join(self.model_path, 'collaborative_model.pkl')
            umm_path = os.path.join(self.model_path, 'user_movie_matrix.pkl')
            if os.path.exists(collab_path) and os.path.exists(umm_path):
                model = joblib.load(collab_path)
                user_movie_matrix = joblib.load(umm_path)

                # SHAP for tree/GBDT models; adjust if your model differs
                self.shap_explainer = shap.TreeExplainer(model)

                # LIME over the user_movie_matrix rows (user feature space)
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    user_movie_matrix.values,
                    feature_names=[f"Movie_{i}" for i in range(user_movie_matrix.shape[1])],
                    class_names=['Dislike', 'Like'],
                    mode='classification'
                )
                return True
        except Exception as e:
            print(f"Error initializing explainers: {e}")
        return False

    def explain_content_recommendation(self, movie_id, recommended_movie_id):
        """Explain why a movie was recommended based on content similarity."""
        try:
            movies_df = joblib.load(os.path.join(self.model_path, 'movies_df.pkl'))
            tfidf_vectorizer = joblib.load(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))

            original_movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
            recommended_movie = movies_df[movies_df['movie_id'] == recommended_movie_id].iloc[0]

            original_text = f"{original_movie['genres']} {original_movie['overview']}"
            recommended_text = f"{recommended_movie['genres']} {recommended_movie['overview']}"

            feature_names = tfidf_vectorizer.get_feature_names_out()
            original_features = tfidf_vectorizer.transform([original_text]).toarray()[0]
            recommended_features = tfidf_vectorizer.transform([recommended_text]).toarray()[0]

            feature_contributions = original_features * recommended_features
            top_features_idx = np.argsort(feature_contributions)[-10:][::-1]

            explanation = {
                'type': 'content_based',
                'features': [
                    {
                        'name': feature_names[idx],
                        'contribution': float(feature_contributions[idx]),
                        'original_value': float(original_features[idx]),
                        'recommended_value': float(recommended_features[idx])
                    }
                    for idx in top_features_idx if feature_contributions[idx] > 0
                ],
                'similarity_factors': {
                    'genres': original_movie['genres'],
                    'vote_average_diff': abs(float(original_movie['vote_average']) - float(recommended_movie['vote_average'])),
                    'popularity_ratio': float(recommended_movie['popularity']) / max(float(original_movie['popularity']), 1.0)
                }
            }
            return explanation

        except Exception as e:
            print(f"Error explaining content recommendation: {e}")
            return {'type': 'content_based', 'features': [], 'error': str(e)}

    def explain_collaborative_recommendation(self, user_id, movie_id):
        """Explain collaborative filtering recommendation using SHAP and LIME (user-global)."""
        if not self.shap_explainer or not self.lime_explainer:
            if not self.initialize_explainers():
                return {'type': 'collaborative', 'error': 'Explainers not available'}

        try:
            user_movie_matrix = joblib.load(os.path.join(self.model_path, 'user_movie_matrix.pkl'))
            model = joblib.load(os.path.join(self.model_path, 'collaborative_model.pkl'))

            if user_id not in user_movie_matrix.index:
                return {'type': 'collaborative', 'error': 'User not found in training data'}

            user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)

            # SHAP for positive class
            shap_values = self.shap_explainer.shap_values(user_ratings)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
            feature_importance = shap_values[0]

            # Top features by absolute contribution
            top_features_idx = np.argsort(np.abs(feature_importance))[-10:][::-1]
            top_shap_vals = feature_importance[top_features_idx]

            # LIME
            lime_exp = self.lime_explainer.explain_instance(
                user_ratings[0],
                model.predict_proba,
                num_features=10
            )

            # Map indices to readable movie titles
            cols = load_user_movie_columns()
            feature_names = [f"Movie_{i}" for i in top_features_idx]
            feature_titles = []
            for fn in feature_names:
                _, t = feature_to_movie(fn, cols)
                feature_titles.append(t or fn)

            explanation = {
                'type': 'collaborative',
                'shap_values': [float(val) for val in top_shap_vals],
                'top_movies': [int(idx) for idx in top_features_idx],
                'feature_names': feature_names,
                'feature_titles': feature_titles,
                'lime_features': [
                    {'feature': exp[0], 'importance': float(exp[1])}
                    for exp in lime_exp.as_list()
                ],
                'prediction_probability': float(model.predict_proba(user_ratings)[0][1])
            }
            return explanation

        except Exception as e:
            print(f"Error explaining collaborative recommendation: {e}")
            return {'type': 'collaborative', 'error': str(e)}

    def _cosine(self, a, b, eps=1e-9):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def explain_item_aware_with_content(self, user_id, candidate_movie_id, content_model=None, top_k=10, rating_baseline=3.0):
        """
        Item-aware contributions for a candidate movie using content embeddings.
        content_model must expose get_embedding(movie_id) -> np.ndarray
        """
        try:
            cm = content_model or getattr(recommendation_system, "content_model", None)
            if cm is None or not hasattr(cm, "get_embedding"):
                return {"error": "no_content_model", "item_contribs": []}

            cand_vec = cm.get_embedding(candidate_movie_id)
            if cand_vec is None:
                return {"error": "no_candidate_embedding", "item_contribs": []}

            # Lazily import to avoid Django registry issues at import time
            from movies.models import Rating, Movie

            user_ratings = list(Rating.objects.filter(user_id=user_id).values_list('movie_id', 'rating'))
            if not user_ratings:
                return {"error": "no_user_ratings", "item_contribs": []}

            contribs = []
            for mid, r in user_ratings:
                v = cm.get_embedding(mid)
                if v is None:
                    continue
                sim = self._cosine(cand_vec, v)
                weight = sim * (float(r) - float(rating_baseline))
                contribs.append((mid, weight))

            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            contribs = contribs[:top_k]

            id_to_title = {m.id: m.title for m in Movie.objects.filter(id__in=[mid for mid, _ in contribs])}
            item_contribs = [{"title": id_to_title.get(mid, f"Movie_{mid}"), "weight": float(w)} for mid, w in contribs]

            return {
                "item_contribs": item_contribs,
                "shap_values": [c["weight"] for c in item_contribs],
                "feature_names": [c["title"] for c in item_contribs],
            }
        except Exception as e:
            print(f"Error in item-aware explanation: {e}")
            return {"error": str(e), "item_contribs": []}

    def generate_explanation_text(self, explanation_data, movie_title):
        """Generate human-readable explanation text."""
        if explanation_data.get('error'):
            return f"Unable to generate explanation: {explanation_data['error']}"

        if explanation_data['type'] == 'content_based':
            features = explanation_data.get('features', [])
            if features:
                top_feature = features[0]['name']
                return f"'{movie_title}' was recommended because it shares similar characteristics with movies you've enjoyed, particularly in terms of '{top_feature}' and other content features."
            return f"'{movie_title}' was recommended based on content similarity with your previously rated movies."

        elif explanation_data['type'] == 'collaborative':
            prob = explanation_data.get('prediction_probability', 0.0)
            return f"'{movie_title}' was recommended with {prob:.1%} confidence based on preferences of users similar to you who also enjoyed movies in your rating history."

        return f"'{movie_title}' was recommended based on its popularity and general appeal."

    def create_shap_plot(self, shap_values, feature_names, max_display=10, sort_values=True):
        """
        Create SHAP-style bar plot as base64 image.
        - Expects shap_values and feature_names aligned.
        - If sort_values is False, preserves the input order (useful for pre-ranked lists).
        """
        try:
            shap_values = np.array(shap_values).astype(float).flatten()
            feature_names = list(feature_names)

            # Trim to max_display
            if len(shap_values) > max_display:
                if sort_values:
                    idx = np.argsort(np.abs(shap_values))[-max_display:]
                    shap_values = shap_values[idx]
                    feature_names = [feature_names[i] for i in idx]
                else:
                    shap_values = shap_values[:max_display]
                    feature_names = feature_names[:max_display]

            # Sort if requested
            if sort_values:
                order = np.argsort(np.abs(shap_values))
                shap_values = shap_values[order]
                feature_names = [feature_names[i] for i in order]

            plt.figure(figsize=(10, 6))
            colors = ['red' if val < 0 else 'blue' for val in shap_values]
            y = range(len(shap_values))
            plt.barh(y, shap_values, color=colors)
            plt.yticks(y, feature_names)
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Feature Importance for Recommendation')
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error creating SHAP plot: {e}")
            return None


# Singleton
explainer = RecommendationExplainer()