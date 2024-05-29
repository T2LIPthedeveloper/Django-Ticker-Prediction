from rest_framework import serializers

# Serializers define the API representation.

class PredictRequestSerializer(serializers.Serializer):
    start_date = serializers.DateField()
    end_date = serializers.DateField()

class PredictResponseSerializer(serializers.Serializer):
    actual_gdp = serializers.ListField(child=serializers.FloatField())
    predicted_gdp_1q = serializers.ListField(child=serializers.FloatField())
    predicted_gdp_2q = serializers.ListField(child=serializers.FloatField())
    predicted_gdp_4q = serializers.ListField(child=serializers.FloatField())
    actual_phase = serializers.ListField(child=serializers.CharField())
    predicted_phase_1q = serializers.ListField(child=serializers.CharField())
    predicted_phase_2q = serializers.ListField(child=serializers.CharField())
    predicted_phase_4q = serializers.ListField(child=serializers.CharField())
