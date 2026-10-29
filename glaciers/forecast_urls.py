
from django.urls import path

from . import forecast_views

app_name = "glacier_forecast"

urlpatterns = [
    path(
        "",
        forecast_views.forecast_dashboard,
        name="dashboard",
    ),
    path(
        "api/",
        forecast_views.forecast_data_api,
        name="data_api",
    ),
]
