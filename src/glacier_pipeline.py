# -*- coding: utf-8 -*-
"""
glacier_pipeline.py \u2014 Glacier Segmentation Pipeline

Orchestrates the end-to-end glacier extent segmentation workflow for
Samudra Tapu Glacier (2018–2026).  Takes existing per-year NDSI candidate
masks (produced by src/ndsi_mask.py) and refines them into authoritative
binary glacier masks using:

  1. GLIMS polygon rasterization (via src/glims_utils.py)
  2. NDSI–GLIMS intersection  (Candidate_Mask)
  3. Image-based refinement   (morphological closing + connected-component filter)
  4. Visual validation output (4-panel PNG per year)
  5. Area summary CSV         (glacier_area_summary.csv)

This module must never modify src/ndsi_mask.py or any existing file under
data/processed/ndsi/.

Usage
-----
    python src/glacier_pipeline.py [--year YEAR] [--ndsi-dir PATH]
                                   [--min-area-km2 FLOAT]
"""

from __future__ import annotations

import argparse
import logging
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from glims_utils import download_glims, load_and_filter_glims

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask operations
# ---------------------------------------------------------------------------

def intersect_masks(ndsi_mask: np.ndarray, glims_mask: np.ndarray) -> np.ndarray:
    """Produce the Candidate_Mask by logically ANDing NDSI and GLIMS masks.

    Rules (applied in priority order):
    - output == 255  where ``ndsi_mask == 255``  (preserve nodata)
    - output == 1    where ``ndsi_mask == 1`` AND ``glims_mask == 1``
    - output == 0    in all other cases

    Parameters
    ----------
    ndsi_mask:
        uint8 array from ndsi_mask.py (values: 0, 1, 255).
    glims_mask:
        uint8 array rasterized from the GLIMS polygon (values: 0, 1).
        Must have the same shape as *ndsi_mask*.

    Returns
    -------
    np.ndarray
        uint8 array containing only the values {0, 1, 255}.
    """
    # Start with zeros, same shape as input
    result = np.zeros(ndsi_mask.shape, dtype=np.uint8)

    # Where both masks are 1 → glacier candidate
    result[(ndsi_mask == 1) & (glims_mask == 1)] = 1

    # Preserve nodata (highest priority, overwrites the glacier-candidate pass)
    result[ndsi_mask == 255] = 255

    return result


# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------

def refine_mask(
    candidate: np.ndarray,
    glims_mask: np.ndarray,
    closing_radius: int = 3,
    min_area_px: int = 10000,
) -> np.ndarray:
    """Apply morphological closing and connected-component filtering.

    Steps (Requirements 4.1 – 4.7):
    1. Convert *candidate* to a boolean working array (True where == 1).
    2. If *closing_radius* > 0, apply binary closing with a circular disk
       kernel of the given radius.  The result is then masked to the GLIMS
       boundary so that no new pixels are introduced outside it (Req 4.4, 4.5).
    3. Label connected components with :func:`scipy.ndimage.label`.
    4. Remove components whose pixel count is strictly less than *min_area_px*
       (Req 4.2).
    5. Return uint8 array containing only {0, 1} — nodata pixels from the
       candidate are treated as 0 (Req 4.6).

    The disk structuring element is a boolean square array of size
    ``(2*closing_radius+1) × (2*closing_radius+1)`` where pixel (i, j)
    (0-indexed, centre at radius) is True when ``(i-r)² + (j-r)² ≤ r²``.

    Parameters
    ----------
    candidate:
        uint8 array from :func:`intersect_masks` (values: 0, 1, 255).
    glims_mask:
        uint8 array (values: 0, 1) with the same shape as *candidate*.
    closing_radius:
        Radius in pixels for the disk structuring element.  0 disables
        morphological closing.
    min_area_px:
        Minimum connected-component size in pixels.  Components smaller than
        this are removed (Req 4.2).

    Returns
    -------
    np.ndarray
        uint8 array with only values {0, 1}.  Nodata (255) pixels from the
        input are mapped to 0.
    """
    # Step 1 — boolean working array; nodata (255) treated as non-glacier
    working = candidate == 1

    # Step 2 — morphological closing, clipped to GLIMS boundary
    if closing_radius > 0:
        r = closing_radius
        size = 2 * r + 1
        # Disk structuring element: True where (i-r)² + (j-r)² ≤ r²
        ii, jj = np.ogrid[:size, :size]
        struct = (ii - r) ** 2 + (jj - r) ** 2 <= r ** 2

        closed = ndimage.binary_closing(working, structure=struct)

        # Mask result to GLIMS boundary (Req 4.4, 4.5)
        glims_boundary = glims_mask == 1
        working = closed & glims_boundary

    # Step 3 — connected-component labelling (Req 4.1)
    labeled_array, num_components = ndimage.label(working)
    logger.debug("Connected components found: %d", num_components)

    # Step 4 — remove small components (Req 4.2)
    if num_components > 0:
        component_sizes = ndimage.sum(
            working, labeled_array, range(1, num_components + 1)
        )
        removed_count = 0
        for label_id, size in enumerate(component_sizes, start=1):
            if size < min_area_px:
                working[labeled_array == label_id] = False
                removed_count += 1
    else:
        removed_count = 0

    final_pixel_count = int(np.sum(working))
    logger.info(
        "refine_mask: components found=%d, removed=%d (< %d px), "
        "final pixel count=%d",
        num_components,
        removed_count,
        min_area_px,
        final_pixel_count,
    )

    # Step 5 — return uint8 {0, 1} (Req 4.6)
    return working.astype(np.uint8)


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def save_raster(
    array: np.ndarray,
    reference_profile: dict,
    out_path: Path,
    band_desc: str = "",
    nodata: int = 255,
) -> None:
    """Write *array* as a uint8 GeoTIFF with DEFLATE compression.

    CRS and affine transform are copied from *reference_profile*.  The nodata
    tag is set to *nodata*.  If *band_desc* is non-empty it is stored as the
    band description for band 1.  Parent directories are created automatically.

    Parameters
    ----------
    array:
        2-D uint8 numpy array to write.
    reference_profile:
        rasterio-style profile dict that must contain at least the keys
        ``crs`` and ``transform`` (width / height are derived from *array*).
    out_path:
        Destination path for the GeoTIFF.  Parent directories are created
        if they do not already exist.
    band_desc:
        Optional band description string stored in the GeoTIFF metadata.
        Ignored when empty.
    nodata:
        Nodata value to record in the file header (default 255).

    Returns
    -------
    None

    Requirements: 2.3, 3.3, 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Ensure parent directory exists (Req 5.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": array.shape[1],
        "height": array.shape[0],
        "crs": reference_profile["crs"],
        "transform": reference_profile["transform"],
        "compress": "deflate",
        "nodata": nodata,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8), 1)
        # Set band description if provided (Req 5.4)
        if band_desc:
            dst.set_band_description(1, band_desc)

    logger.debug("Saved raster to %s (nodata=%d, band_desc=%r)", out_path, nodata, band_desc)


# ---------------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------------

def compute_area_km2(mask: np.ndarray, pixel_res_m: float = 10.0) -> float:
    """Return the glacier area in km² for a binary mask.

    Each pixel where *mask* == 1 contributes ``pixel_res_m²`` square metres.
    The total is converted to km² by dividing by 1 × 10⁶.

    Parameters
    ----------
    mask:
        uint8 array with values {0, 1} (or {0, 1, 255}).  Only pixels equal
        to 1 are counted as glacier (Req 7.1).
    pixel_res_m:
        Ground sampling distance in metres.  Defaults to 10.0 m (Sentinel-2
        10 m bands), giving 100 m² per pixel.

    Returns
    -------
    float
        Glacier area in km².
    """
    return float(np.sum(mask == 1) * pixel_res_m ** 2 / 1e6)


# ---------------------------------------------------------------------------
# Visual validation
# ---------------------------------------------------------------------------

def make_validation_plot(
    year: int,
    rgb_path: Path,
    ndsi_mask: np.ndarray,
    glims_mask: np.ndarray,
    candidate: np.ndarray,
    final: np.ndarray,
    area_km2: float,
    out_path: Path,
) -> None:
    """Render and save a 4-panel validation figure for a single year.

    Panels
    ------
    (a) True-colour RGB composite from Sentinel-2 bands B4/B3/B2 with a
        2–98 % percentile stretch per channel.
    (b) NDSI_Mask as a binary image with the GLIMS polygon boundary contoured
        in red.
    (c) Candidate_Mask (NDSI ∩ GLIMS).
    (d) Final_Mask after morphological closing and connected-component filtering.

    Parameters
    ----------
    year:
        The processing year shown in the figure title.
    rgb_path:
        Path to the multiband Sentinel-2 GeoTIFF; bands are read in order,
        and the first three are treated as B4, B3, B2.
    ndsi_mask:
        uint8 array (values: 0, 1, 255) from ``ndsi_mask.py``.
    glims_mask:
        uint8 array (values: 0, 1) rasterized from the GLIMS polygon.
    candidate:
        uint8 array (values: 0, 1, 255) — the Candidate_Mask.
    final:
        uint8 array (values: 0, 1) — the Final_Mask.
    area_km2:
        Final glacier area in km², shown in the figure title.
    out_path:
        Destination path for the PNG output.  Parent directories are
        created automatically.

    Returns
    -------
    None

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend — no GUI window (Req 6.6)
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------ #
    # Panel (a): True-colour RGB from raw Sentinel-2 raster              #
    # ------------------------------------------------------------------ #
    with rasterio.open(rgb_path) as src:
        # Read the first three bands (B4, B3, B2 for Sentinel-2)
        b4 = src.read(1).astype(np.float32)
        b3 = src.read(2).astype(np.float32)
        b2 = src.read(3).astype(np.float32)

    def _stretch(band: np.ndarray) -> np.ndarray:
        """Apply 2–98 % percentile stretch and return a 0–1 float32 array."""
        lo, hi = np.percentile(band[np.isfinite(band)], [2, 98])
        if hi == lo:
            return np.zeros_like(band, dtype=np.float32)
        stretched = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        return stretched.astype(np.float32)

    rgb = np.stack([_stretch(b4), _stretch(b3), _stretch(b2)], axis=-1)

    # ------------------------------------------------------------------ #
    # Figure layout                                                        #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # (a) True-colour RGB
    axes[0].imshow(rgb)
    axes[0].set_title("(a) RGB Composite")
    axes[0].axis("off")

    # (b) NDSI_Mask with GLIMS boundary contour
    axes[1].imshow(ndsi_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].contour(glims_mask == 1, levels=[0.5], colors="red", linewidths=0.8)
    axes[1].set_title("(b) NDSI Mask + GLIMS boundary")
    axes[1].axis("off")

    # (c) Candidate_Mask
    axes[2].imshow(candidate, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("(c) Candidate Mask")
    axes[2].axis("off")

    # (d) Final_Mask
    axes[3].imshow(final, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("(d) Final Mask")
    axes[3].axis("off")

    # Figure title (Req 6.4)
    fig.suptitle(
        f"Samudra Tapu Glacier {year} \u2014 {area_km2:.2f} km\u00b2",
        fontsize=14,
        y=1.01,
    )

    # Save and release (Req 6.1, 6.5)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Validation plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

def process_year(
    year: int,
    ndsi_dir: Path,
    raw_dir: Path,
    glims_dir: Path,
    out_dirs: dict,
    glims_gdf,          # gpd.GeoDataFrame
    closing_radius: int,
    min_area_px: int,
) -> "dict | None":
    """Orchestrate all pipeline stages for a single year.

    Stage sequence
    --------------
    1. Rasterize the GLIMS polygon to the NDSI grid.
    2. Load the NDSI mask and GLIMS mask arrays from disk.
    3. Verify spatial alignment (shape, CRS, transform).
    4. Compute the Candidate_Mask (NDSI ∩ GLIMS).
    5. Save the Candidate_Mask.
    6. Refine the Candidate_Mask (morphological closing + component filter).
    7. Save the Final_Mask.
    8. Compute pixel counts and areas.
    9. Log per-year progress.
    10. Generate the 4-panel validation PNG.

    Parameters
    ----------
    year:
        Calendar year to process (e.g. 2020).
    ndsi_dir:
        Directory containing ``Samudra_Tapu_YEAR_Scientific_mask.tif`` and
        ``Samudra_Tapu_YEAR_Scientific_ndsi.tif``.
    raw_dir:
        Directory containing ``Samudra_Tapu_YEAR_Scientific.tif`` (multiband
        Sentinel-2 raw raster for the RGB validation panel).
    glims_dir:
        Directory used for intermediate GLIMS mask outputs (passed through to
        ``rasterize_glims``).
    out_dirs:
        Mapping of output-type keys to ``Path`` objects:
        ``{"glims": Path, "candidates": Path, "final_masks": Path,
           "validation": Path}``.
    glims_gdf:
        Filtered ``gpd.GeoDataFrame`` for Samudra Tapu Glacier (obtained once
        by the caller and re-used across years to avoid repeated I/O).
    closing_radius:
        Structuring-element radius (pixels) for morphological closing.
        0 disables closing.
    min_area_px:
        Minimum connected-component size in pixels.  Components smaller than
        this are removed from the Final_Mask.

    Returns
    -------
    dict or None
        Stats dictionary on success::

            {
                "year": int,
                "candidate_pixels": int,
                "candidate_area_km2": float,
                "final_pixels": int,
                "final_area_km2": float,
            }

        Returns ``None`` if the NDSI mask file for *year* does not exist
        (logs a WARNING in that case).

    Requirements: 3.5, 5.1–5.5, 6.1–6.6, 7.1–7.3, 8.5, 8.6
    """
    from glims_utils import rasterize_glims  # local import avoids circular deps

    # ------------------------------------------------------------------ #
    # Resolve file paths                                                  #
    # ------------------------------------------------------------------ #
    ndsi_mask_path = ndsi_dir / f"Samudra_Tapu_{year}_Scientific_mask.tif"
    rgb_path = raw_dir / f"Samudra_Tapu_{year}_Scientific.tif"

    glims_mask_path = out_dirs["glims"] / f"glims_mask_{year}.tif"
    candidate_path = out_dirs["candidates"] / f"glacier_candidate_{year}.tif"
    final_path = out_dirs["final_masks"] / f"glacier_mask_{year}.tif"
    validation_path = out_dirs["validation"] / f"validation_{year}.png"

    # ------------------------------------------------------------------ #
    # Stage: check NDSI input exists (Req 8.5)                           #
    # ------------------------------------------------------------------ #
    if not ndsi_mask_path.exists():
        logger.warning(
            "NDSI mask not found for year %d — skipping. (Expected: %s)",
            year,
            ndsi_mask_path,
        )
        return None

    # ------------------------------------------------------------------ #
    # Stage 1: Rasterize GLIMS polygon to NDSI grid                      #
    # ------------------------------------------------------------------ #
    rasterize_glims(glims_gdf, ndsi_mask_path, glims_mask_path)

    # ------------------------------------------------------------------ #
    # Stage 2: Load arrays from disk                                      #
    # ------------------------------------------------------------------ #
    with rasterio.open(ndsi_mask_path) as src:
        ndsi_array = src.read(1).astype(np.uint8)
        ndsi_profile = dict(src.profile)
        ndsi_crs = src.crs
        ndsi_transform = src.transform
        ndsi_shape = (src.height, src.width)

    with rasterio.open(glims_mask_path) as src:
        glims_array = src.read(1).astype(np.uint8)
        glims_crs = src.crs
        glims_transform = src.transform
        glims_shape = (src.height, src.width)

    # ------------------------------------------------------------------ #
    # Stage 3: Alignment guard (Req 2.4)                                  #
    # ------------------------------------------------------------------ #
    if ndsi_shape != glims_shape:
        raise ValueError(
            f"Alignment error for year {year}: "
            f"NDSI mask shape {ndsi_shape} != GLIMS mask shape {glims_shape}."
        )
    if ndsi_crs != glims_crs:
        raise ValueError(
            f"Alignment error for year {year}: "
            f"NDSI CRS {ndsi_crs} != GLIMS CRS {glims_crs}."
        )
    if ndsi_transform != glims_transform:
        raise ValueError(
            f"Alignment error for year {year}: "
            f"NDSI transform {ndsi_transform} != GLIMS transform {glims_transform}."
        )

    # ------------------------------------------------------------------ #
    # Stage 4: Intersect → Candidate_Mask                                #
    # ------------------------------------------------------------------ #
    candidate = intersect_masks(ndsi_array, glims_array)

    # ------------------------------------------------------------------ #
    # Stage 5: Save Candidate_Mask                                        #
    # ------------------------------------------------------------------ #
    save_raster(
        candidate,
        ndsi_profile,
        candidate_path,
        band_desc="candidate_mask",
        nodata=255,
    )

    # ------------------------------------------------------------------ #
    # Stage 6: Refine → Final_Mask                                        #
    # ------------------------------------------------------------------ #
    final = refine_mask(candidate, glims_array, closing_radius, min_area_px)

    # ------------------------------------------------------------------ #
    # Stage 7: Save Final_Mask                                            #
    # ------------------------------------------------------------------ #
    save_raster(
        final,
        ndsi_profile,
        final_path,
        band_desc="glacier_mask",
        nodata=255,
    )

    # ------------------------------------------------------------------ #
    # Stage 8: Compute areas                                              #
    # ------------------------------------------------------------------ #
    candidate_pixels: int = int(np.sum(candidate == 1))
    final_pixels: int = int(np.sum(final == 1))
    candidate_area_km2: float = compute_area_km2(candidate)
    final_area_km2: float = compute_area_km2(final)

    # ------------------------------------------------------------------ #
    # Stage 9: Log per-year progress (Req 8.6)                           #
    # ------------------------------------------------------------------ #
    print(
        f"Year {year}: "
        f"candidate area = {candidate_area_km2:.3f} km²  |  "
        f"final area = {final_area_km2:.3f} km²"
    )
    logger.info(
        "process_year %d complete — candidate_px=%d (%.3f km²), "
        "final_px=%d (%.3f km²)",
        year,
        candidate_pixels,
        candidate_area_km2,
        final_pixels,
        final_area_km2,
    )

    # ------------------------------------------------------------------ #
    # Stage 10: Validation plot                                           #
    # ------------------------------------------------------------------ #
    make_validation_plot(
        year=year,
        rgb_path=rgb_path,
        ndsi_mask=ndsi_array,
        glims_mask=glims_array,
        candidate=candidate,
        final=final,
        area_km2=final_area_km2,
        out_path=validation_path,
    )

    return {
        "year": year,
        "candidate_pixels": candidate_pixels,
        "candidate_area_km2": candidate_area_km2,
        "final_pixels": final_pixels,
        "final_area_km2": final_area_km2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Glacier Segmentation Pipeline")
    parser.add_argument("--year", type=int, help="Process a single year instead of 2018-2026")
    parser.add_argument("--ndsi-dir", type=Path, default=Path("data/processed/ndsi"), help="Directory containing NDSI masks")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw Sentinel-2 data")
    parser.add_argument("--min-area-km2", type=float, default=1.0, help="Minimum connected component area in km²")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    years = [args.year] if args.year else list(range(2018, 2027))
    ndsi_dir = args.ndsi_dir
    raw_dir = args.raw_dir

    glims_dir = Path("data/glims")
    out_dirs = {
        "glims": Path("data/processed/glims"),
        "candidates": Path("data/processed/candidates"),
        "final_masks": Path("data/processed/final_masks"),
        "validation": Path("data/processed/validation"),
    }

    # Fetch and filter GLIMS data
    geojson_path = download_glims(glims_dir)
    glims_gdf = load_and_filter_glims(geojson_path)

    # 1 km² = 1,000,000 m² = 10,000 pixels at 10m x 10m
    min_area_px = int(args.min_area_km2 * 1_000_000 / (10.0 * 10.0))

    results = []
    for year in years:
        result = process_year(
            year=year,
            ndsi_dir=ndsi_dir,
            raw_dir=raw_dir,
            glims_dir=glims_dir,
            out_dirs=out_dirs,
            glims_gdf=glims_gdf,
            closing_radius=3,
            min_area_px=min_area_px,
        )
        if result is not None:
            results.append(result)

    if not results:
        logger.error("No years were successfully processed.")
        sys.exit(1)

    results.sort(key=lambda x: x["year"])

    summary_path = Path("data/processed/glacier_area_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "candidate_pixels", "candidate_area_km2", "final_pixels", "final_area_km2"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("Pipeline complete. Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
