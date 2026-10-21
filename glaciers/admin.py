from django.contrib import admin
from .models import Glacier, AreaObservation


@admin.register(Glacier)
class GlacierAdmin(admin.ModelAdmin):
    list_display = ("name", "glims_id", "reference_area_km2")
    search_fields = ("name", "glims_id")


@admin.register(AreaObservation)
class AreaObservationAdmin(admin.ModelAdmin):
    list_display = (
        "glacier",
        "year",
        "area_km2",
        "glacier_pixels",
        "validation_status",
    )
    list_filter = ("year", "validation_status")
    search_fields = ("glacier__name",)