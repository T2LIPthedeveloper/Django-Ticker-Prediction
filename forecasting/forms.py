from django import forms
from .models import Forecast

class ForecastForm(forms.ModelForm):
    class Meta:
        model = Forecast
        fields = ['quarters_ahead']
