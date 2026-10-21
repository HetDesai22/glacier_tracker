from rest_framework import serializers
from .models import Glacier

class GlacierSerializer(serializers.ModelSerializer):
    class Meta:
        model = Glacier
        fields = [
            "id",
            "name",
            "glims_id",
            "latitude",
            "longitude",
            "reference_area_km2",
        ]
        

from .models import AreaObservation


class AreaObservationSerializer(serializers.ModelSerializer):
    class Meta:
        model = AreaObservation
        fields = [
            "year",
            "area_km2",
            "glacier_pixels",
            "source",
            "validation_status",
        ]
