
import csv
from pathlib import Path

from django.conf import settings


PROJECT_ROOT = Path(settings.BASE_DIR)

METRICS_CSV = PROJECT_ROOT / "evaluation" / "metrics_by_year.csv"
FORECAST_CSV = (
    PROJECT_ROOT
    / "evaluation"
    / "forecasts"
    / "samudra_tapu_forecast.csv"
)


class ForecastDataError(Exception):
    """Raised when historical or forecast data is unavailable or invalid."""


def _read_csv(path):
    if not path.is_file():
        raise ForecastDataError(f"Required data file not found: {path}")

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error) as exc:
        raise ForecastDataError(f"Could not read {path.name}: {exc}") from exc


def _number(value, field, year):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ForecastDataError(
            f"Invalid {field} value for year {year}: {value!r}"
        )

    if not result == result or abs(result) == float("inf"):
        raise ForecastDataError(
            f"Non-finite {field} value for year {year}"
        )

    return result


def load_glacier_forecast():
    """Load historical areas and forecast values for the dashboard."""

    historical_rows = _read_csv(METRICS_CSV)
    forecast_rows = _read_csv(FORECAST_CSV)

    historical = []
    forecast = []

    seen_years = set()

    for row in historical_rows:
        if row.get("status") != "OK":
            continue

        try:
            year = int(row["year"])
        except (TypeError, ValueError, KeyError):
            raise ForecastDataError("Historical CSV contains an invalid year.")

        if year in seen_years:
            raise ForecastDataError(f"Duplicate historical year: {year}")

        area = _number(row.get("prediction_area_km2"), "historical area", year)

        if area < 0:
            raise ForecastDataError(f"Negative historical area for {year}")

        seen_years.add(year)

        historical.append({
            "year": year,
            "area_km2": area,
            "type": "historical",
            "source": "U-Net prediction",
            "dice": _number(row["dice"], "Dice", year)
            if row.get("dice") not in (None, "") else None,
            "iou": _number(row["iou"], "IoU", year)
            if row.get("iou") not in (None, "") else None,
        })

    if not historical:
        raise ForecastDataError("No successful historical observations found.")

    historical.sort(key=lambda item: item["year"])
    last_observed_year = historical[-1]["year"]

    forecast_seen_years = set()

    for row in forecast_rows:
        try:
            year = int(row["year"])
        except (TypeError, ValueError, KeyError):
            raise ForecastDataError("Forecast CSV contains an invalid year.")

        if year in forecast_seen_years or year in seen_years:
            raise ForecastDataError(f"Duplicate year in forecast data: {year}")

        area = _number(
            row.get("predicted_area_km2"), "forecast area", year
        )

        if area < 0:
            raise ForecastDataError(
                f"Forecast for {year} is negative. Review the model output."
            )

        forecast_seen_years.add(year)

        forecast.append({
            "year": year,
            "area_km2": area,
            "type": "forecast",
            "source": "Experimental linear trend",
            "dice": None,
            "iou": None,
        })

    forecast.sort(key=lambda item: item["year"])

    if forecast and forecast[0]["year"] <= last_observed_year:
        raise ForecastDataError(
            "Forecast years must come after the latest observed year."
        )

    combined = historical + forecast

    # Changes are calculated between consecutive mapped-area values.
    # Forecast-period changes remain projections, not observed glacier change.
    previous = None

    for item in combined:
        item["change_km2"] = (
            None if previous is None
            else item["area_km2"] - previous["area_km2"]
        )

        if item["type"] == "historical":
            item["change_type"] = (
                "historical_comparison" if previous is not None else None
            )
        else:
            item["change_type"] = "forecast_comparison"

        previous = item

    return {
        "historical": historical,
        "forecast": forecast,
        "combined": combined,
        "last_observed_year": last_observed_year,
        "forecast_start_year": forecast[0]["year"] if forecast else None,
        "forecast_end_year": forecast[-1]["year"] if forecast else None,
        "historical_count": len(historical),
        "forecast_count": len(forecast),
    }
