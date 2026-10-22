from django.urls import path
from .views import GlacierListAPIView, GlacierAreaListAPIView, dashboard

urlpatterns = [
    path("glaciers/", GlacierListAPIView.as_view(), name="glacier-list"),
    
    path( "glaciers/<int:glacier_id>/areas/", GlacierAreaListAPIView.as_view(), name="glacier-area-list"),
    
    path("", dashboard, name="dashboard"),
]