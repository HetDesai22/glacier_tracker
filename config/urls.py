from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="/forecast/", permanent=False), name="home"),
    path("admin/", admin.site.urls),

    # Keep your existing URL patterns here.
    path("forecast/", include("glaciers.forecast_urls")),
]