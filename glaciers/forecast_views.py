from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .forecast_service import ForecastDataError, load_glacier_forecast


def forecast_dashboard(request):
    try:
        data = load_glacier_forecast()
    except ForecastDataError as exc:
        return render(
            request,
            "glaciers/forecast_dashboard.html",
            {
                "error": str(exc),
                "data": None,
            },
            status=503,
        )

    return render(
        request,
        "glaciers/forecast_dashboard.html",
        {
            "data": data,
            "error": None,
        },
    )


@require_GET
def forecast_data_api(request):
    try:
        data = load_glacier_forecast()
    except ForecastDataError as exc:
        return JsonResponse(
            {
                "error": "Forecast data is unavailable.",
                "detail": str(exc),
            },
            status=503,
        )

    return JsonResponse(data)
