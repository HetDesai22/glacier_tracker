
import rasterio
import numpy as np
from pathlib import Path

root = Path(__file__).parent

for year in [2024, 2026]:
    files = {
        "Predicted": root / "data" / "processed" / "final_masks" / f"glacier_mask_{year}.tif",
        "GLIMS": root / "data" / "processed" / "glims" / f"glims_mask_{year}.tif",
    }

    print(f"\n{'=' * 20} {year} {'=' * 20}")

    for label, path in files.items():
        with rasterio.open(path) as src:
            data = src.read(1)
            values, counts = np.unique(data, return_counts=True)

            print(f"\n{label}: {path.name}")
            print("Shape:", data.shape)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("NoData metadata:", src.nodata)
            print("Pixel values:", dict(zip(values.tolist(), counts.tolist())))
            print("Bounds:", src.bounds)

    with rasterio.open(files["Predicted"]) as p, \
         rasterio.open(files["GLIMS"]) as g:
        print("\nSame shape:", p.shape == g.shape)
        print("Same CRS:", p.crs == g.crs)
        print("Same transform:", p.transform == g.transform)
        print("Same bounds:", p.bounds == g.bounds)
