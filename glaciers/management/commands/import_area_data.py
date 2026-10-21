
import csv
from pathlib import Path
import math

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from glaciers.models import Glacier, AreaObservation


class Command(BaseCommand):
    help = "Import annual glacier area observations from a CSV file."

    def add_arguments(self, parser):
        parser.add_argument(
            "--glims-id",
            required=True,
            help="GLIMS ID of the glacier already registered in the database.",
        )

    def handle(self, *args, **options):
        glims_id = options["glims_id"]

        # Locate the CSV relative to the project root.
        csv_path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "outputs"
            / "glacier_area_by_year.csv"
        )

        if not csv_path.is_file():
            raise CommandError(f"CSV file not found: {csv_path}")

        # Read and validate the CSV structure.
        required_columns = {
            "year",
            "glacier_pixels",
            "pixel_area_m2",
            "glacier_area_km2",
        }

        with csv_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames or not required_columns.issubset(
                reader.fieldnames
            ):
                raise CommandError(
                    f"CSV must contain: {sorted(required_columns)}"
                )

            rows = list(reader)

        if not rows:
            raise CommandError("CSV contains no area observations.")

        # Convert values and validate the calculations.
        validated_rows = []
        seen_years = set()

        try:
            for row in rows:
                year = int(row["year"])
                pixels = int(row["glacier_pixels"])
                pixel_area = float(row["pixel_area_m2"])
                area = float(row["glacier_area_km2"])

                if not all(
                    math.isfinite(value)
                    for value in (pixel_area, area)
                ):
                    raise ValueError(
                        f"Non-finite area value for year {year}"
                    )

                if year < 1:
                    raise ValueError(f"Invalid year: {year}")

                if year in seen_years:
                    raise ValueError(f"Duplicate year: {year}")
                seen_years.add(year)

                if pixels < 0 or pixel_area <= 0 or area < 0:
                    raise ValueError(
                        f"Invalid area values for year {year}"
                    )

                expected_area = pixels * pixel_area / 1_000_000

                if abs(expected_area - area) > 0.0001:
                    raise ValueError(
                        f"Area mismatch for {year}: "
                        f"CSV={area}, calculated={expected_area}"
                    )

                validated_rows.append({
                    "year": year,
                    "glacier_pixels": pixels,
                    "area_km2": area,
                })

        except (ValueError, TypeError, KeyError) as exc:
            raise CommandError(f"Invalid CSV data: {exc}") from exc

        # Find the glacier already registered in PostgreSQL.
        try:
            glacier = Glacier.objects.get(glims_id=glims_id)
        except Glacier.DoesNotExist as exc:
            raise CommandError(
                f"No glacier registered with GLIMS ID {glims_id}. "
                "Register the glacier in Django Admin first."
            ) from exc

        created_count = 0
        updated_count = 0
        unchanged_count = 0

        # Import all observations as one database transaction.
        with transaction.atomic():
            for item in validated_rows:
                observation, created = (
                    AreaObservation.objects.get_or_create(
                        glacier=glacier,
                        year=item["year"],
                        defaults={
                            "area_km2": item["area_km2"],
                            "glacier_pixels": item["glacier_pixels"],
                            "source": "Satellite segmentation",
                            "validation_status": "pending",
                        },
                    )
                )

                if created:
                    created_count += 1
                    continue

                changed = (
                    observation.area_km2 != item["area_km2"]
                    or observation.glacier_pixels
                    != item["glacier_pixels"]
                    or observation.source
                    != "Satellite segmentation"
                )

                if changed:
                    observation.area_km2 = item["area_km2"]
                    observation.glacier_pixels = item["glacier_pixels"]
                    observation.source = "Satellite segmentation"
                    observation.save(
                        update_fields=[
                            "area_km2",
                            "glacier_pixels",
                            "source",
                        ]
                    )
                    updated_count += 1
                else:
                    unchanged_count += 1

        self.stdout.write(self.style.SUCCESS("Import completed."))
        self.stdout.write(f"Glacier: {glacier.name} ({glacier.glims_id})")
        self.stdout.write(f"Rows validated: {len(validated_rows)}")
        self.stdout.write(f"Created: {created_count}")
        self.stdout.write(f"Updated: {updated_count}")
        self.stdout.write(f"Unchanged: {unchanged_count}")
