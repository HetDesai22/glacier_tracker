from rest_framework import generics
from .models import Glacier, AreaObservation
from .serializers import GlacierSerializer, AreaObservationSerializer
from django.shortcuts import render


class GlacierListAPIView(generics.ListAPIView):
    queryset = Glacier.objects.all().order_by("name")
    serializer_class = GlacierSerializer
    
class GlacierAreaListAPIView(generics.ListAPIView):
    serializer_class = AreaObservationSerializer
    
    def get_queryset(self):
        return AreaObservation.objects.filter(
            glacier_id=self.kwargs["glacier_id"]
        ).order_by("year")
        
def dashboard(request):
    return render(request, "glaciers/dashboard.html")


