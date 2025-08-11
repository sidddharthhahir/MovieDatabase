from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from movies.models import Rating

User = get_user_model()

class Command(BaseCommand):
    help = "Delete synthetic users and ratings"

    def handle(self, *args, **opts):
        # Delete by email pattern (safer than username)
        ratings_deleted = Rating.objects.filter(user__email__iendswith='@example.com').delete()[0]
        users_deleted = User.objects.filter(email__iendswith='@example.com').delete()[0]
        
        self.stdout.write(f"Deleted {ratings_deleted} ratings and {users_deleted} users")