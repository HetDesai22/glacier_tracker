"""Create glacier (snow/ice) masks from Sentinel-2 GeoTIFFs using NDSI.

NDSI = (Green - SWIR1) / (Green + SWIR1) = (B3 - B11) / (B3 + B11)
A pixel is classified as snow/ice when NDSI > threshold (default 0.4).

Outputs per input image (in data/processed/ndsi/):
  <name>_ndsi.tif      float32 NDSI values (nodata = NaN)
  <name>_mask.tif      uint8 mask: 1 = snow/ice, 0 = other, 255 = nodata
  <name>_preview.png   RGB with the mask outlined in cyan
and a summary CSV with snow/ice area per image.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
MASK_NODATA = 255


def band_index(src, name):
    """1-based index of a band by its description (e.g. 'B3')."""
    try:
        return list(src.descriptions).index(name) + 1
    except ValueError:
        raise ValueError(f"{src.name}: band {name} not found in {src.descriptions}")


def compute_ndsi(green, swir):
    green = green.astype(np.float32)
    swir = swir.astype(np.float32)
    denom = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = (green - swir) / denom
    ndsi[denom == 0] = np.nan
    return ndsi


def save_preview(src, mask, valid, out_path, title):
    rgb = np.stack([src.read(band_index(src, b)) for b in ("B4", "B3", "B2")], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb[valid], (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb)
    axes[0].contour(mask == 1, levels=[0.5], colors="cyan", linewidths=0.6)
    axes[0].set_title("RGB + NDSI mask outline")
    cmap = matplotlib.colors.ListedColormap(["#3b3b3b", "#e6f4ff"])
    cmap.set_bad("black")
    axes[1].imshow(np.where(valid, mask, np.nan), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Snow/ice mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def process(path, out_dir, threshold, preview):
    with rasterio.open(path) as src:
        green = src.read(band_index(src, "B3"))
        swir = src.read(band_index(src, "B11"))
        valid = (green > 0) & (swir > 0)

        ndsi = compute_ndsi(green, swir)
        ndsi[~valid] = np.nan

        mask = np.zeros(ndsi.shape, dtype=np.uint8)
        mask[ndsi > threshold] = 1
        mask[~valid] = MASK_NODATA

        profile = src.profile.copy()
        profile.update(count=1, compress="deflate")

        stem = path.stem
        with rasterio.open(out_dir / f"{stem}_ndsi.tif", "w", **{**profile, "dtype": "float32", "nodata": np.nan}) as dst:
            dst.write(ndsi, 1)
            dst.set_band_description(1, "NDSI")
        with rasterio.open(out_dir / f"{stem}_mask.tif", "w", **{**profile, "dtype": "uint8", "nodata": MASK_NODATA}) as dst:
            dst.write(mask, 1)
            dst.set_band_description(1, f"NDSI>{threshold}")

        if preview:
            save_preview(src, mask, valid, out_dir / f"{stem}_preview.png", f"{stem}  (NDSI > {threshold})")

        pixel_area_km2 = abs(src.res[0] * src.res[1]) / 1e6
        ice_px = int((mask == 1).sum())
        valid_px = int(valid.sum())
        return {
            "image": path.name,
            "ice_pixels": ice_px,
            "valid_pixels": valid_px,
            "ice_area_km2": round(ice_px * pixel_area_km2, 4),
            "ice_fraction": round(ice_px / valid_px, 4) if valid_px else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "ndsi")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--no-preview", action="store_true", help="skip PNG previews")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.input_dir.glob("*.tif"))
    if not images:
        raise SystemExit(f"No .tif files found in {args.input_dir}")

    rows = []
    for path in images:
        row = process(path, args.output_dir, args.threshold, not args.no_preview)
        rows.append(row)
        print(f"{row['image']}: {row['ice_area_km2']:.2f} km² snow/ice ({row['ice_fraction']:.1%} of valid pixels)")

    summary = args.output_dir / "ndsi_summary.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote masks and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
