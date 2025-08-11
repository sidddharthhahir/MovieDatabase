from django.db import models
from django.contrib.auth import get_user_model
from movies.models import Movie

User = get_user_model()

class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    score = models.FloatField()
    algorithm = models.CharField(max_length=50)  # 'collaborative', 'content', 'hybrid'
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'movie', 'algorithm')
        ordering = ['-score']
    
    def __str__(self):
        return f"{self.user.username} - {self.movie.title}: {self.score}"

class Explanation(models.Model):
    recommendation = models.OneToOneField(Recommendation, on_delete=models.CASCADE)
    shap_values = models.JSONField()
    lime_explanation = models.JSONField()
    feature_importance = models.JSONField()
    explanation_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Explanation for {self.recommendation}"