from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import ForecastForm
from .ltc_model import predict_macro_phase

def index(request):
    if request.method == "POST":
        form = ForecastForm(request.POST)
        if form.is_valid():
            quarters_ahead = form.cleaned_data['quarters_ahead']
            result = predict_macro_phase(quarters_ahead)
            return render(request, 'forecasting/result.html', {'result': result})
    else:
        form = ForecastForm()
    return render(request, 'forecasting/index.html', {'form': form})
