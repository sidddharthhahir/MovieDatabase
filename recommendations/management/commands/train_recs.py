from django.core.management.base import BaseCommand
from recommendations.ml_models import recommendation_system

class Command(BaseCommand):
    help = "Train/refresh recommendation models (content and collaborative)."

    def handle(self, *args, **kwargs):
        try:
            recommendation_system.train_content_model()
            recommendation_system.train_collaborative_model()
            self.stdout.write(self.style.SUCCESS("Models trained successfully."))
        except Exception as e:
            self.stderr.write(f"Error: {e}")