from django.db import models

# Create your models here.
class Forecast(models.Model):
    quarters_ahead = models.IntegerField(choices=[(1, '1'), (2, '2'), (3, '3'), (4, '4')])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Forecast for {self.quarters_ahead} quarter(s) ahead"