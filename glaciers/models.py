
from django.db import models


class Glacier(models.Model):
    name = models.CharField(max_length=200)
    glims_id = models.CharField(max_length=50, unique=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    reference_area_km2 = models.FloatField(
        null=True,
        blank=True,
        help_text="Area from the reference glacier inventory, not an annual measurement.",
    )

    def __str__(self):
        return self.name


class AreaObservation(models.Model):
    glacier = models.ForeignKey(
        Glacier,
        on_delete=models.CASCADE,
        related_name="area_observations",
    )
    year = models.PositiveSmallIntegerField()
    area_km2 = models.FloatField()
    glacier_pixels = models.PositiveIntegerField()
    source = models.CharField(max_length=100, default="Satellite segmentation")
    validation_status = models.CharField(
        max_length=30,
        default="pending",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["glacier", "year"],
                name="unique_glacier_area_year",
            ),
        ]
        ordering = ["year"]

    def __str__(self):
        return f"{self.glacier.name} — {self.year}"
