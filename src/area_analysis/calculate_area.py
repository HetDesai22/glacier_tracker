import rasterio
import numpy as np
from pathlib import Path
import csv

mask_dir = Path("data/processed/final_masks")

results = []

for year in range(2018, 2027):
    mask_path = mask_dir / f"glacier_mask_{year}.tif"

    if not mask_path.exists():
        print(f"Skipping {year}: mask file not found")
        continue

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

        glacier_pixels = np.count_nonzero(mask == 1)

        pixel_area = abs(src.res[0] * src.res[1])
        area_km2 = glacier_pixels * pixel_area / 1_000_000
        
        results.append({
            "year": year,
            "glacier_pixels": glacier_pixels,
            "pixel_area_m2": pixel_area,
            "glacier_area_km2": area_km2,
        })
    
output_path = Path("data/outputs/glacier_area_by_year.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Saved {len(results)} records to {output_path}")