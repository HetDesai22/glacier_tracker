from pathlib import Path
import numpy as np
import rasterio

mask_dir = Path("data/processed/final_masks")
expected_years = range(2018, 2027)
errors = []

for year in expected_years:
    path = mask_dir / f"glacier_mask_{year}.tif"

    if not path.exists():
        print(f"MISSING: {year}")
        errors.append(year)
        continue

    try:
        with rasterio.open(path) as src:
            mask = src.read(1)
            values, counts = np.unique(mask, return_counts=True)

            print(f"\nYear: {year}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"CRS: {src.crs}")
            print(f"Pixel values/counts: {dict(zip(values.tolist(), counts.tolist()))}")
            print(f"Valid data type: {src.dtypes[0]}")
            print(f"File readable: YES")

            if not set(values.tolist()).issubset({0, 1, 255}):
                print("WARNING: Unexpected mask values")
            if 1 not in values:
                print("WARNING: No glacier pixels classified")
            if src.width == 0 or src.height == 0:
                errors.append(year)

    except Exception as exc:
        print(f"ERROR reading {year}: {exc}")
        errors.append(year)

print("\n" + "=" * 45)
print(f"Masks checked: {len(list(expected_years))}")
print(f"Errors: {errors if errors else 'None detected'}")
